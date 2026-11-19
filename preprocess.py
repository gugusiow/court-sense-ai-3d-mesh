"""Build a two-view Neural Body dataset from a synchronized practice clip.

The loop, for every paired frame, is:

1. Undistort both images with the shared-world calibration.
2. YOLO finds the player and the basketball in each view.
3. SAM 2 turns those boxes into masks. The ball mask is subtracted from the
   player mask, with a small halo, so the body NeRF never sees the ball.
4. SAM 3D Body consumes the full frame, the player box, the ball-free mask,
   and the real intrinsics. It returns an MHR mesh. The mesh is transformed
   into the calibration world and fused across the two views.
5. The ball center is triangulated from the two ball masks and saved beside
   the body, not inside it.

Run ``python preprocess.py --self-check`` before downloading model weights.
That path uses a synthetic player and ball and only needs OpenCV and NumPy.

The training environment is separate. Neural Body's official install is
Python 3.7 and PyTorch 1.4, and it does not run on this Windows workstation.
See docs/nvs_two_view.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from courtsense.nvs.export import DatasetWriter, FrameRecord  # noqa: E402
from courtsense.nvs.geometry import (  # noqa: E402
    FrameUndistorter,
    baseline_meters,
    camera_vertices_to_world,
    ensure_frame_size,
    fuse_world_meshes,
    load_cameras,
    mesh_confidence,
    optical_axis_angle_deg,
    select_pair,
    vertices_in_camera,
)
from courtsense.nvs.pipeline import (  # noqa: E402
    PreprocessOptions,
    detection_record,
    process_pair_images,
)
from courtsense.nvs.synthetic import (  # noqa: E402
    SyntheticDetector,
    SyntheticMesher,
    SyntheticSegmenter,
    make_practice_scene,
)


def main() -> None:
    args = _parse_args()
    if args.self_check:
        _self_check(Path(args.output), args.overwrite)
        return
    if args.mesh_from_export:
        _mesh_from_export(args)
        return
    _detection_pass(args)


def _detection_pass(args: argparse.Namespace) -> None:
    import cv2

    cameras = select_pair(load_cameras(args.cameras))
    undistorters = [FrameUndistorter(camera) for camera in cameras]
    export_cameras = [item.camera for item in undistorters]
    _report_rig(export_cameras[0], export_cameras[1])
    options = _options(args)
    detector_back, detector_side = _build_detectors(args)
    segmenter = None if args.skip_segment else _build_segmenter(args)
    mesher = None if args.skip_mesh else _build_mesher(args)
    writer = DatasetWriter(Path(args.output), export_cameras, overwrite=args.overwrite)

    back_cap = _open_video(args.video_back)
    side_cap = _open_video(args.video_side)
    _warn_if_fps_differs(back_cap, side_cap, cv2)
    _skip_frames(back_cap, args.start_frame)
    _skip_frames(side_cap, args.start_frame)

    source_index = args.start_frame
    export_index = 0
    skipped = 0
    ended = False
    try:
        while not ended and (args.max_frames is None or export_index < args.max_frames):
            back = _read_one(back_cap, cv2)
            side = _read_one(side_cap, cv2)
            if back is None or side is None:
                break
            ensure_frame_size(back, cameras[0])
            ensure_frame_size(side, cameras[1])
            back = undistorters[0].apply(back)
            side = undistorters[1].apply(side)
            result = process_pair_images(
                back,
                side,
                export_cameras[0],
                export_cameras[1],
                detector_back,
                detector_side,
                segmenter,
                mesher,
                options,
            )
            if result.back.player is None and result.side.player is None:
                skipped += 1
            else:
                record = _record(export_index, source_index, result)
                writer.write_views(
                    record,
                    [back, side],
                    [result.back.player_mask, result.side.player_mask],
                    [result.back.ball_mask, result.side.ball_mask],
                    [detection_record(result.back), detection_record(result.side)],
                    result.ball,
                )
                writer.write_meshes(
                    export_index,
                    result.fused_vertices,
                    result.per_view_payloads,
                    result.faces,
                    result.fusion_mode,
                    result.mesh_rmse_m,
                    {"back": result.back.confidence, "side": result.side.confidence},
                )
                _print_frame(record, result)
                export_index += 1
            source_index += 1
            for _ in range(args.frame_stride - 1):
                if not back_cap.grab() or not side_cap.grab():
                    ended = True
                    break
                source_index += 1
    finally:
        back_cap.release()
        side_cap.release()

    summary = writer.close(
        {
            "skipped_no_player": skipped,
            "mesh_ran": mesher is not None,
            "segmenter": "yolo_box" if segmenter is None else "sam2",
            "mesh_translation": options.mesh_translation,
        }
    )
    _print_summary(summary, Path(args.output))


def _mesh_from_export(args: argparse.Namespace) -> None:
    """Second pass: SAM 3D Body only, reading masks already on disk.

    SAM 2 and SAM 3D Body together can exceed a 16 GB GPU. Run the first
    pass with ``--skip-mesh``, then this pass in a fresh process.
    """
    import cv2

    output = Path(args.output)
    cameras = select_pair(load_cameras(output / "cameras_export.json"))
    options = _options(args)
    mesher = _build_mesher(args)
    writer = DatasetWriter(output, cameras, resume=True)
    for record in list(writer.records):
        views = []
        confidences = []
        meshes = []
        payloads: list[dict | None] = []
        for camera_index, camera in enumerate(cameras):
            image = cv2.imread(str(output / "images" / str(camera_index) / f"{record.export_index:06d}.jpg"))
            if image is None:
                raise FileNotFoundError(f"missing image for export frame {record.export_index}")
            mask_path = output / "mask" / str(camera_index) / f"{record.export_index:06d}.png"
            player_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            player_mask = np.zeros(image.shape[:2], dtype=bool) if player_mask is None else player_mask > 0
            detection = json.loads(
                (output / "detections" / str(camera_index) / f"{record.export_index:06d}.json").read_text(
                    encoding="utf-8"
                )
            )
            mesh = None
            world = None
            confidence = 0.0
            if detection["player_xyxy"] is not None and player_mask.any():
                mesh = mesher.estimate(
                    image,
                    np.asarray(detection["player_xyxy"], dtype=np.float64),
                    player_mask,
                    camera,
                )
                if mesh is not None and "pred_vertices" in mesh:
                    confidence = mesh_confidence(mesh, image.shape[:2])
                    world = camera_vertices_to_world(
                        vertices_in_camera(mesh, options.mesh_translation),
                        camera,
                    )
                    meshes.append(world)
                    confidences.append(max(confidence, 0.01))
            views.append(confidence)
            payloads.append(_mesh_payload(mesh, world, confidence, detection["mask_source"]))
        if meshes:
            fused, mode, rmse = fuse_world_meshes(meshes, confidences, options.mesh_agreement_m)
            rmse_value = None if not np.isfinite(rmse) else float(rmse)
        else:
            fused, mode, rmse_value = None, "skipped", None
        names = [camera.view_name for camera in cameras]
        writer.write_meshes(
            record.export_index,
            fused,
            payloads,
            getattr(mesher, "faces", None),
            mode,
            rmse_value,
            {names[0]: views[0], names[1]: views[1]},
        )
        print(f"[{record.export_index:04d}] mesh {mode} rmse={rmse_value}")
    summary = writer.close({"mesh_ran": True, "mesh_from_export": True})
    _print_summary(summary, output)


def _self_check(output: Path, overwrite: bool) -> None:
    cameras, world_vertices, ball_world, player_box = make_practice_scene()
    back, side = cameras
    options = PreprocessOptions(ball_dilation_px=2, mesh_agreement_m=0.05)
    mesher = SyntheticMesher(cameras, world_vertices)
    segmenter = SyntheticSegmenter()
    writer = DatasetWriter(output, cameras, overwrite=overwrite)
    frame = np.zeros((back.height, back.width, 3), dtype=np.uint8)
    for index in range(3):
        result = process_pair_images(
            frame,
            frame,
            back,
            side,
            SyntheticDetector(back, ball_world, player_box),
            SyntheticDetector(side, ball_world, player_box),
            segmenter,
            mesher,
            options,
        )
        record = _record(index, index, result)
        writer.write_views(
            record,
            [frame, frame],
            [result.back.player_mask, result.side.player_mask],
            [result.back.ball_mask, result.side.ball_mask],
            [detection_record(result.back), detection_record(result.side)],
            result.ball,
        )
        writer.write_meshes(
            index,
            result.fused_vertices,
            result.per_view_payloads,
            result.faces,
            result.fusion_mode,
            result.mesh_rmse_m,
            {"back": result.back.confidence, "side": result.side.confidence},
        )
    summary = writer.close({"self_check": True})
    fused = np.load(output / "vertices" / "000000.npy")
    if summary["fusion_average"] != 3 or summary["ball_triangulated"] != 3:
        raise RuntimeError(f"self-check failed: {summary}")
    if np.max(np.abs(fused - world_vertices)) > 1e-4:
        raise RuntimeError("self-check mesh does not match the synthetic world points")
    print(f"self-check passed: {output}")
    print(json.dumps(summary, indent=2))


def _record(export_index: int, source_index: int, result) -> FrameRecord:
    ball = result.ball
    radius = float(ball["radius_m"])
    reprojection = float(ball["reprojection_px"])
    return FrameRecord(
        export_index=export_index,
        source_frame_index=source_index,
        fusion_mode=result.fusion_mode,
        mesh_rmse_m=result.mesh_rmse_m,
        view_confidence={"back": result.back.confidence, "side": result.side.confidence},
        ball_valid=bool(ball["valid"]),
        ball_reprojection_px=None if not np.isfinite(reprojection) else reprojection,
        radius_m=None if not np.isfinite(radius) else radius,
        radius_clamped=bool(ball["radius_clamped"]),
        mask_source={"back": result.back.mask_source, "side": result.side.mask_source},
    )


def _mesh_payload(mesh, world, confidence: float, mask_source: str) -> dict | None:
    if mesh is None and world is None:
        return None
    payload: dict = {
        "confidence": np.float64(confidence),
        "mask_source": np.asarray(mask_source),
    }
    if world is not None:
        payload["vertices_world"] = np.asarray(world, dtype=np.float32)
    if mesh is not None:
        for key, value in mesh.items():
            if value is None:
                continue
            payload[key] = np.asarray(value)
    return payload


def _print_frame(record: FrameRecord, result) -> None:
    rmse = "n/a" if record.mesh_rmse_m is None else f"{record.mesh_rmse_m:.3f}m"
    ball = "yes" if record.ball_valid else "no"
    errors = [view.mesh_error for view in (result.back, result.side) if view.mesh_error]
    suffix = f" mesh_error={errors[0]}" if errors else ""
    print(
        f"[{record.export_index:04d}] source={record.source_frame_index} "
        f"fusion={record.fusion_mode} rmse={rmse} ball={ball}{suffix}"
    )


def _print_summary(summary: dict, output: Path) -> None:
    print(json.dumps(summary, indent=2))
    if summary["frames_exported"] and summary["fusion_best_view"] > summary["fusion_average"]:
        print(
            "The two MHR meshes disagree on most frames. Check that both cameras "
            "share one world frame and that --mesh-translation matches the checkpoint."
        )
    if summary["frames_exported"] and summary["ball_triangulated"] == 0:
        print("No ball was triangulated. Lower --ball-confidence or check that YOLO's sports-ball class fires.")
    print(f"dataset: {output}")
    print("Next: docs/nvs_two_view.md, section 'Train Neural Body'.")


def _report_rig(back, side) -> None:
    angle = optical_axis_angle_deg(back, side)
    baseline = baseline_meters(back, side)
    print(f"optical-axis angle {angle:.1f} deg, baseline {baseline:.2f} m")
    if angle < 20.0:
        print("The optical axes are close. Ball depth from two views will be noisy.")


def _options(args: argparse.Namespace) -> PreprocessOptions:
    return PreprocessOptions(
        ball_dilation_px=args.ball_dilation,
        ball_box_padding_px=args.ball_box_padding,
        mesh_agreement_m=args.mesh_agreement_m,
        mesh_translation=args.mesh_translation,
    )


def _build_detectors(args: argparse.Namespace):
    from courtsense.nvs.backends import YoloDetector

    weights = args.yolo_weights
    return (
        YoloDetector(weights, args.person_confidence, args.ball_confidence, args.yolo_device),
        YoloDetector(weights, args.person_confidence, args.ball_confidence, args.yolo_device),
    )


def _build_segmenter(args: argparse.Namespace):
    from courtsense.nvs.backends import Sam2Segmenter

    if not args.sam2_checkpoint:
        raise SystemExit("--sam2-checkpoint is required unless you pass --skip-segment")
    return Sam2Segmenter(
        args.sam2_config,
        args.sam2_checkpoint,
        args.device,
        args.sam2_repo,
    )


def _build_mesher(args: argparse.Namespace):
    from courtsense.nvs.backends import Sam3DBodyEstimator

    if not args.sam3d_checkpoint or not args.mhr_path:
        raise SystemExit("--sam3d-checkpoint and --mhr-path are required unless you pass --skip-mesh")
    return Sam3DBodyEstimator(
        args.sam3d_checkpoint,
        args.mhr_path,
        args.sam3d_repo,
        args.inference_type,
    )


def _open_video(path: str):
    import cv2

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    return capture


def _warn_if_fps_differs(back_cap, side_cap, cv2) -> None:
    back_fps = float(back_cap.get(cv2.CAP_PROP_FPS) or 0.0)
    side_fps = float(side_cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if back_fps > 0 and side_fps > 0 and abs(back_fps - side_fps) > 0.5:
        print(f"Video frame rates differ ({back_fps:.2f} vs {side_fps:.2f}). Pairing by frame index will drift.")


def _skip_frames(capture, count: int) -> None:
    for _ in range(count):
        if not capture.grab():
            raise SystemExit("video ended before --start-frame")


def _read_one(capture, cv2):
    ok, frame = capture.read()
    if not ok:
        return None
    return _as_bgr(frame, cv2)


def _as_bgr(frame: np.ndarray, cv2) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess two synchronized views for a mesh-guided NeRF.")
    parser.add_argument("--video-back", help="Back-view video.")
    parser.add_argument("--video-side", help="Side-view video, synchronized by frame index.")
    parser.add_argument("--cameras", help="Shared-world camera JSON. See config/cameras.example.json.")
    parser.add_argument("--output", default="data/nvs/shot", help="Export directory.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-check", action="store_true", help="Write a tiny synthetic dataset and check the geometry.")
    parser.add_argument("--skip-segment", action="store_true", help="Use YOLO rectangles instead of SAM 2.")
    parser.add_argument("--skip-mesh", action="store_true", help="Export images, masks, and the ball. Run the mesh later.")
    parser.add_argument("--mesh-from-export", action="store_true", help="Run SAM 3D Body on an export produced with --skip-mesh.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after this many exported frames.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--yolo-weights", default="yolo11m.pt")
    parser.add_argument("--yolo-device", default="cpu")
    parser.add_argument("--person-confidence", type=float, default=0.35)
    parser.add_argument("--ball-confidence", type=float, default=0.15)
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--sam2-checkpoint", default="")
    parser.add_argument("--sam2-repo", default="third_party/sam2")
    parser.add_argument("--sam3d-checkpoint", default="")
    parser.add_argument("--mhr-path", default="")
    parser.add_argument("--sam3d-repo", default="third_party/sam-3d-body")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-type", default="full", choices=("full", "body"))
    parser.add_argument("--ball-dilation", type=int, default=8, help="Pixels removed around the ball in the player mask.")
    parser.add_argument("--ball-box-padding", type=int, default=4)
    parser.add_argument("--mesh-agreement-m", type=float, default=0.08)
    parser.add_argument(
        "--mesh-translation",
        default="add_pred_cam_t",
        choices=("add_pred_cam_t", "vertices_are_camera_frame"),
        help="Official SAM 3D Body visualizer uses pred_vertices + pred_cam_t.",
    )
    args = parser.parse_args()
    if args.self_check and args.output == "data/nvs/shot":
        args.output = "data/nvs/self_check"
    if args.frame_stride < 1:
        parser.error("--frame-stride must be >= 1")
    if not args.self_check and not args.mesh_from_export:
        if not args.video_back or not args.video_side or not args.cameras:
            parser.error("--video-back, --video-side, and --cameras are required")
    return args


if __name__ == "__main__":
    main()
