import json
from pathlib import Path

import numpy as np
import pytest

from courtsense.nvs.calibration import (
    checkerboard_object_points,
    pose_from_corners,
    select_shared_board_pose,
)
from courtsense.nvs.composite import composite_ball
from courtsense.nvs.export import DatasetWriter, build_annots, save_annots
from courtsense.nvs.geometry import (
    Camera,
    ball_disk_mask,
    camera_vertices_to_world,
    fuse_world_meshes,
    load_cameras,
    look_at_w2c,
    project_points,
    select_pair,
    separate_player_and_ball,
    triangulate_point,
    vertices_in_camera,
)
from courtsense.nvs.pipeline import PreprocessOptions, detection_record, process_pair_images
from courtsense.nvs.synthetic import (
    SyntheticDetector,
    SyntheticMesher,
    SyntheticSegmenter,
    make_practice_scene,
)

ROOT = Path(__file__).resolve().parents[2]


def _camera(width, height, eye, target, focal=None) -> Camera:
    focal = float(width) if focal is None else focal
    rotation, translation = look_at_w2c(np.asarray(eye, dtype=np.float64), np.asarray(target, dtype=np.float64))
    intrinsic = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return Camera(
        "cam",
        "view",
        width,
        height,
        intrinsic,
        rotation,
        translation,
        np.zeros(5),
    )


def test_look_at_places_the_camera_center_and_the_principal_point():
    eye = np.array([0.2, -3.0, 1.5])
    target = np.array([0.2, 0.0, 1.2])
    camera = _camera(200, 100, eye, target, focal=150.0)
    assert np.allclose(camera.center_world(), eye)
    rotation, _ = look_at_w2c(eye, target)
    point = eye + 2.0 * rotation[2]
    pixels, depth = project_points(camera, point)
    assert depth[0] > 0
    assert np.allclose(pixels[0], [100.0, 50.0], atol=1e-6)


def test_triangulation_recovers_a_world_point():
    target = np.array([0.0, 0.0, 1.2])
    back = _camera(320, 240, [0.0, -4.0, 1.7], target, focal=280.0)
    side = _camera(320, 240, [4.0, 0.0, 1.7], target, focal=280.0)
    point = np.array([0.3, -0.2, 1.4])
    back_uv, back_depth = project_points(back, point)
    side_uv, side_depth = project_points(side, point)
    assert back_depth[0] > 0 and side_depth[0] > 0
    recovered = triangulate_point(back, back_uv[0], side, side_uv[0])
    assert np.allclose(recovered, point, atol=1e-6)


def test_ball_pixels_leave_the_player_mask_and_stay_in_the_ball_mask():
    player = np.ones((30, 30), dtype=bool)
    ball = np.zeros((30, 30), dtype=bool)
    ball[12:18, 14:20] = True
    separated, saved_ball = separate_player_and_ball(player, ball, dilation_px=2)
    assert np.array_equal(saved_ball, ball)
    assert not separated[14, 16]
    assert separated[1, 1]
    assert not separated[12, 14]


def test_large_mesh_disagreement_keeps_the_more_confident_view():
    confident = np.zeros((4, 3))
    other = np.ones((4, 3))
    fused, mode, rmse = fuse_world_meshes([confident, other], [0.9, 0.2], agreement_m=0.08)
    assert mode == "best_view"
    assert rmse > 0.08
    assert np.allclose(fused, confident)


def test_agreeing_meshes_are_averaged_in_world_space():
    base = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])
    other = base + np.array([0.01, 0.0, 0.0])
    fused, mode, rmse = fuse_world_meshes([base, other], [1.0, 1.0], agreement_m=0.08)
    assert mode == "average"
    assert rmse < 0.08
    assert np.allclose(fused, base + np.array([0.005, 0.0, 0.0]))


def test_camera_translation_is_added_before_the_world_transform():
    camera = _camera(64, 48, [0.0, -2.0, 1.0], [0.0, 0.0, 1.0], focal=40.0)
    world = np.array([[0.0, 0.0, 1.0]])
    camera_points = (camera.R @ world.T).T + camera.t
    person = {"pred_vertices": camera_points - np.array([0.2, -0.1, 0.4]), "pred_cam_t": np.array([0.2, -0.1, 0.4])}
    restored = camera_vertices_to_world(vertices_in_camera(person, "add_pred_cam_t"), camera)
    assert np.allclose(restored, world)


def test_annots_translation_is_millimeters_and_paths_use_forward_slashes():
    cameras, _, _, _ = make_practice_scene()
    annots = build_annots(cameras, frame_count=2)
    translation = np.asarray(annots["cams"]["T"][0], dtype=np.float64)
    assert translation.shape == (3, 1)
    assert np.allclose(translation.reshape(3) / 1000.0, cameras[0].t)
    assert annots["ims"][1]["ims"] == ["images/0/000001.jpg", "images/1/000001.jpg"]
    assert np.allclose(annots["cams"]["D"][0], 0.0)


def test_example_camera_file_is_a_proper_rotation():
    cameras = load_cameras(ROOT / "config" / "cameras.example.json")
    back, side = select_pair(cameras)
    assert back.view_name == "back"
    assert side.view_name == "side"
    assert abs(np.linalg.det(back.R) - 1.0) < 1e-6
    assert abs(np.linalg.det(side.R) - 1.0) < 1e-6
    assert np.allclose(back.center_world(), [0.0, -6.0, 1.6], atol=1e-6)
    assert np.allclose(side.center_world(), [6.0, 0.0, 1.6], atol=1e-6)


def test_depth_in_front_of_the_ball_hides_it():
    intrinsic = np.array([[50.0, 0.0, 20.0], [0.0, 50.0, 20.0], [0.0, 0.0, 1.0]])
    camera = Camera("cam", "view", 40, 40, intrinsic, np.eye(3), np.zeros(3), np.zeros(5))
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    center = np.array([0.0, 0.0, 5.0])
    painted = composite_ball(image, camera, center, radius_m=0.5)
    assert painted[20, 20, 2] > 100
    hidden = composite_ball(image, camera, center, radius_m=0.5, depth_m=np.ones((40, 40)))
    assert hidden[20, 20, 2] == 0
    empty_depth = composite_ball(image, camera, center, radius_m=0.5, depth_m=np.zeros((40, 40)))
    assert empty_depth[20, 20, 2] > 100
    assert ball_disk_mask(camera, center, 0.5)[20, 20]
    assert not ball_disk_mask(camera, center, 0.5)[20, 0]


def test_synthetic_export_keeps_the_ball_out_of_the_body_mask(tmp_path):
    pytest.importorskip("cv2")
    cameras, world_vertices, ball_world, player_box = make_practice_scene()
    back, side = cameras
    options = PreprocessOptions(ball_dilation_px=2, mesh_agreement_m=0.05)
    frame = np.zeros((back.height, back.width, 3), dtype=np.uint8)
    result = process_pair_images(
        frame,
        frame,
        back,
        side,
        SyntheticDetector(back, ball_world, player_box),
        SyntheticDetector(side, ball_world, player_box),
        SyntheticSegmenter(),
        SyntheticMesher(cameras, world_vertices),
        options,
    )
    assert result.fusion_mode == "average"
    assert np.allclose(result.fused_vertices, world_vertices, atol=1e-6)
    assert bool(result.ball["valid"])
    assert np.linalg.norm(result.ball["center_world"] - ball_world) < 0.05

    writer = DatasetWriter(tmp_path / "export", cameras, overwrite=True)
    from courtsense.nvs.export import FrameRecord

    record = FrameRecord(
        0,
        0,
        result.fusion_mode,
        result.mesh_rmse_m,
        {"back": result.back.confidence, "side": result.side.confidence},
        True,
        float(result.ball["reprojection_px"]),
        float(result.ball["radius_m"]),
        bool(result.ball["radius_clamped"]),
        {"back": result.back.mask_source, "side": result.side.mask_source},
    )
    writer.write_views(
        record,
        [frame, frame],
        [result.back.player_mask, result.side.player_mask],
        [result.back.ball_mask, result.side.ball_mask],
        [detection_record(result.back), detection_record(result.side)],
        result.ball,
    )
    writer.write_meshes(
        0,
        result.fused_vertices,
        result.per_view_payloads,
        result.faces,
        result.fusion_mode,
        result.mesh_rmse_m,
        record.view_confidence,
    )
    summary = writer.close()
    assert summary["ball_triangulated"] == 1
    save_annots(tmp_path / "export" / "annots.npy", cameras, 1)
    loaded = np.load(tmp_path / "export" / "annots.npy", allow_pickle=True).item()
    assert loaded["ims"][0]["ims"][0] == "images/0/000000.jpg"

    import cv2

    player_mask = cv2.imread(str(tmp_path / "export" / "mask" / "0" / "000000.png"), cv2.IMREAD_GRAYSCALE)
    ball_u, ball_v = result.ball["back_uv"]
    assert player_mask[int(round(ball_v)), int(round(ball_u))] == 0
    assert int(player_mask.sum()) > 0
    metadata = json.loads((tmp_path / "export" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["format"] == "neuralbody-zju-mocap-v1+mhr"
    assert (tmp_path / "export" / "mask_ball" / "1" / "000000.png").exists()
    assert (tmp_path / "export" / "vertices" / "000000.ply").exists()


def test_checkerboard_pose_roundtrips_and_rejects_a_flipped_board():
    pytest.importorskip("cv2")
    object_points = checkerboard_object_points(5, 4, 0.04)
    target = object_points.mean(axis=0)
    width, height, focal = 640, 480, 600.0
    back = _camera(width, height, [target[0], target[1] - 0.9, 0.35], target, focal=focal)
    side = _camera(width, height, [target[0] + 0.9, target[1], 0.35], target, focal=focal)
    back_uv, back_depth = project_points(back, object_points)
    side_uv, side_depth = project_points(side, object_points)
    assert np.all(back_depth > 0) and np.all(side_depth > 0)
    assert np.all((back_uv[:, 0] > 1) & (back_uv[:, 0] < width - 1))
    assert np.all((side_uv[:, 0] > 1) & (side_uv[:, 0] < width - 1))
    rotation, translation, error = pose_from_corners(object_points, back_uv, back.K, np.zeros(5))
    assert error < 1e-3
    assert np.allclose(rotation, back.R, atol=1e-4)
    assert np.allclose(translation, back.t, atol=1e-4)
    chosen = select_shared_board_pose(
        object_points,
        [(back_uv, side_uv)],
        back.K,
        np.zeros(5),
        side.K,
        np.zeros(5),
        (width, height),
        (width, height),
    )
    assert chosen["corner_order"] == "as_detected"
    assert chosen["alignment_rmse_m"] < 1e-4
    assert np.allclose(chosen["side_t"], side.t, atol=1e-4)
