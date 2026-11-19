"""Per-frame body, ball, and mask assembly for one synchronized pair.

The heavy models are injected. This module only decides which pixels belong
to the body, which belong to the ball, and where the shared-world mesh sits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from courtsense.nvs.geometry import (
    BoxDetection,
    Camera,
    ball_disk_mask,
    basketball_radius_m,
    box_mask,
    camera_vertices_to_world,
    fuse_world_meshes,
    mask_centroid,
    mesh_confidence,
    pad_box,
    project_points,
    separate_player_and_ball,
    triangulate_point,
    undistort_pixel,
    vertices_in_camera,
)


# FIBA size-7 circumference is 749–780 mm, so the radius is about 0.119–0.124 m.
SIZE_7_RADIUS_M = 0.121


@dataclass(frozen=True)
class PreprocessOptions:
    ball_dilation_px: int = 8
    ball_box_padding_px: int = 4
    mesh_agreement_m: float = 0.08
    mesh_translation: str = "add_pred_cam_t"
    basketball_radius_m: float = SIZE_7_RADIUS_M
    radius_min_m: float = 0.09
    radius_max_m: float = 0.16
    max_reprojection_px: float = 30.0
    max_ball_distance_m: float = 20.0


@dataclass
class ViewObservation:
    view_name: str
    player: BoxDetection | None
    ball: BoxDetection | None
    player_mask: np.ndarray
    ball_mask: np.ndarray
    mask_source: str
    mesh: dict | None
    mesh_error: str | None
    confidence: float
    world_vertices: np.ndarray | None


@dataclass
class PairResult:
    back: ViewObservation
    side: ViewObservation
    fused_vertices: np.ndarray | None
    fusion_mode: str
    mesh_rmse_m: float | None
    faces: np.ndarray | None
    ball: dict
    per_view_payloads: list[dict | None]


class PersonDetector:
    def detect(self, frame_bgr: np.ndarray) -> tuple[BoxDetection | None, BoxDetection | None]:
        raise NotImplementedError


class ForegroundSegmenter:
    def segment(
        self,
        frame_bgr: np.ndarray,
        player_box: np.ndarray | None,
        ball_box: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        raise NotImplementedError


class BodyMeshEstimator:
    faces: np.ndarray | None

    def estimate(
        self,
        frame_bgr: np.ndarray,
        player_box: np.ndarray,
        player_mask: np.ndarray,
        camera: Camera,
    ) -> dict | None:
        raise NotImplementedError


def observe_view(
    frame_bgr: np.ndarray,
    camera: Camera,
    detector: PersonDetector,
    segmenter: ForegroundSegmenter | None,
    mesher: BodyMeshEstimator | None,
    options: PreprocessOptions,
) -> ViewObservation:
    height, width = frame_bgr.shape[:2]
    player, ball = detector.detect(frame_bgr)
    player_mask = np.zeros((height, width), dtype=bool)
    ball_mask = np.zeros((height, width), dtype=bool)
    mask_source = "empty"
    if player is not None:
        segmented_player = None
        segmented_ball = None
        ball_box = None
        if ball is not None:
            ball_box = pad_box(ball.xyxy, options.ball_box_padding_px, width, height)
        if segmenter is not None:
            segmented_player, segmented_ball = segmenter.segment(
                frame_bgr,
                player.xyxy,
                ball_box,
            )
            mask_source = "sam2"
        if segmented_player is None or not np.any(segmented_player):
            segmented_player = box_mask((height, width), player.xyxy)
            mask_source = "bbox_fallback" if segmenter is not None else "yolo_box"
        if ball is not None and (segmented_ball is None or not np.any(segmented_ball)):
            segmented_ball = box_mask((height, width), ball_box if ball_box is not None else ball.xyxy)
        player_mask, ball_mask = separate_player_and_ball(
            segmented_player,
            segmented_ball,
            options.ball_dilation_px,
        )
    elif ball is not None:
        ball_box = pad_box(ball.xyxy, options.ball_box_padding_px, width, height)
        segmented_ball = None
        if segmenter is not None:
            _, segmented_ball = segmenter.segment(frame_bgr, None, ball_box)
            mask_source = "sam2"
        if segmented_ball is None or not np.any(segmented_ball):
            segmented_ball = box_mask((height, width), ball_box)
            mask_source = "yolo_box"
        ball_mask = segmented_ball.astype(bool)

    mesh = None
    mesh_error = None
    world_vertices = None
    confidence = 0.0
    if mesher is not None and player is not None and np.any(player_mask):
        try:
            # The estimator receives the full undistorted frame, the YOLO box,
            # and the ball-removed mask. It crops internally. A pre-crop would
            # move the principal point away from K, so the two views would no
            # longer meet in the calibration world.
            mesh = mesher.estimate(frame_bgr, player.xyxy, player_mask, camera)
        except Exception as exc:  # noqa: BLE001 - one bad frame should not drop the clip
            mesh_error = f"{type(exc).__name__}: {exc}"
        if mesh is not None and "pred_vertices" in mesh:
            confidence = mesh_confidence(mesh, (height, width))
            camera_vertices = vertices_in_camera(mesh, options.mesh_translation)
            world_vertices = camera_vertices_to_world(camera_vertices, camera)
        elif mesh_error is None and mesh is not None:
            mesh_error = "mesh output did not contain pred_vertices"
            mesh = None
    return ViewObservation(
        view_name=camera.view_name,
        player=player,
        ball=ball,
        player_mask=player_mask,
        ball_mask=ball_mask,
        mask_source=mask_source,
        mesh=mesh,
        mesh_error=mesh_error,
        confidence=confidence,
        world_vertices=world_vertices,
    )


def fuse_pair(
    back: ViewObservation,
    side: ViewObservation,
    back_camera: Camera,
    side_camera: Camera,
    options: PreprocessOptions,
    faces: np.ndarray | None,
) -> PairResult:
    meshes = []
    confidences = []
    if back.world_vertices is not None:
        meshes.append(back.world_vertices)
        confidences.append(max(back.confidence, 0.01))
    if side.world_vertices is not None:
        meshes.append(side.world_vertices)
        confidences.append(max(side.confidence, 0.01))
    if meshes:
        fused, mode, rmse = fuse_world_meshes(meshes, confidences, options.mesh_agreement_m)
    else:
        fused, mode, rmse = None, "skipped", None
    rmse_value = None if rmse is None or not np.isfinite(rmse) else float(rmse)
    ball = triangulate_ball(back, side, back_camera, side_camera, options)
    return PairResult(
        back=back,
        side=side,
        fused_vertices=fused,
        fusion_mode=mode,
        mesh_rmse_m=rmse_value,
        faces=faces,
        ball=ball,
        per_view_payloads=[_view_payload(back), _view_payload(side)],
    )


def triangulate_ball(
    back: ViewObservation,
    side: ViewObservation,
    back_camera: Camera,
    side_camera: Camera,
    options: PreprocessOptions,
) -> dict:
    back_uv = mask_centroid(back.ball_mask, None if back.ball is None else back.ball.xyxy)
    side_uv = mask_centroid(side.ball_mask, None if side.ball is None else side.ball.xyxy)
    center = np.full(3, np.nan, dtype=np.float64)
    raw_radius = np.nan
    radius = np.nan
    clamped = False
    reprojection = np.nan
    valid = False
    reason = "ball missing in a view"
    if back_uv is not None and side_uv is not None and back.ball is not None and side.ball is not None:
        back_pixel = undistort_pixel(back_camera, back_uv)
        side_pixel = undistort_pixel(side_camera, side_uv)
        center = triangulate_point(back_camera, back_pixel, side_camera, side_pixel)
        back_depth = float(project_points(back_camera, center)[1][0])
        side_depth = float(project_points(side_camera, center)[1][0])
        back_reproj, _ = project_points(back_camera, center)
        side_reproj, _ = project_points(side_camera, center)
        reprojection = float(
            np.mean(
                [
                    np.linalg.norm(back_reproj[0] - back_pixel),
                    np.linalg.norm(side_reproj[0] - side_pixel),
                ]
            )
        )
        back_distance = float(np.linalg.norm(center - back_camera.center_world()))
        radius_back = basketball_radius_m(back_camera, back.ball.xyxy, back_depth)
        radius_side = basketball_radius_m(side_camera, side.ball.xyxy, side_depth)
        raw_radius = float(np.mean([radius_back, radius_side]))
        valid = (
            np.isfinite(center).all()
            and back_depth > 0.3
            and side_depth > 0.3
            and 0.3 < back_distance < options.max_ball_distance_m
            and reprojection <= options.max_reprojection_px
        )
        reason = "ok" if valid else "triangulation rejected"
        if valid and options.radius_min_m <= raw_radius <= options.radius_max_m:
            radius = raw_radius
        elif valid:
            radius = options.basketball_radius_m
            clamped = True
            reason = "radius clamped to a size-7 basketball"
        else:
            radius = np.nan
    return {
        "center_world": center.astype(np.float64),
        "radius_m": np.float64(radius),
        "radius_raw_m": np.float64(raw_radius),
        "radius_clamped": np.bool_(clamped),
        "valid": np.bool_(valid),
        "reprojection_px": np.float64(reprojection),
        "back_uv": np.full(2, np.nan) if back_uv is None else back_uv.astype(np.float64),
        "side_uv": np.full(2, np.nan) if side_uv is None else side_uv.astype(np.float64),
        "reason": np.asarray(reason),
    }


def process_pair_images(
    back_bgr: np.ndarray,
    side_bgr: np.ndarray,
    back_camera: Camera,
    side_camera: Camera,
    detector_back: PersonDetector,
    detector_side: PersonDetector,
    segmenter: ForegroundSegmenter | None,
    mesher: BodyMeshEstimator | None,
    options: PreprocessOptions,
) -> PairResult:
    """Run one synchronized pair.

    Detectors are separate objects so a tracker inside one camera cannot
    treat the other camera's player as a teleport.
    """
    back = observe_view(back_bgr, back_camera, detector_back, segmenter, mesher, options)
    side = observe_view(side_bgr, side_camera, detector_side, segmenter, mesher, options)
    faces = None if mesher is None else getattr(mesher, "faces", None)
    return fuse_pair(back, side, back_camera, side_camera, options, faces)


def detection_record(view: ViewObservation) -> dict:
    return {
        "view_name": view.view_name,
        "player_xyxy": None if view.player is None else view.player.xyxy.tolist(),
        "player_confidence": None if view.player is None else view.player.confidence,
        "player_class": None if view.player is None else view.player.class_name,
        "ball_xyxy": None if view.ball is None else view.ball.xyxy.tolist(),
        "ball_confidence": None if view.ball is None else view.ball.confidence,
        "mask_source": view.mask_source,
        "mesh_error": view.mesh_error,
    }


def _view_payload(view: ViewObservation) -> dict | None:
    if view.mesh is None and view.world_vertices is None:
        return None
    payload: dict = {
        "confidence": np.float64(view.confidence),
        "mask_source": np.asarray(view.mask_source),
    }
    if view.world_vertices is not None:
        payload["vertices_world"] = np.asarray(view.world_vertices, dtype=np.float32)
    if view.mesh is not None:
        for key, value in view.mesh.items():
            if key == "mask" or value is None:
                continue
            payload[key] = np.asarray(value)
    return payload


def ball_visible_mask(camera: Camera, ball: dict) -> np.ndarray:
    if not bool(ball["valid"]):
        return np.zeros((camera.height, camera.width), dtype=bool)
    return ball_disk_mask(camera, ball["center_world"], float(ball["radius_m"]))
