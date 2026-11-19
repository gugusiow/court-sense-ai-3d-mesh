"""Cameras, triangulation, mask separation, and MHR mesh fusion.

Camera convention throughout this package is OpenCV's:

- camera axes are x right, y down, z forward
- extrinsics are world-to-camera, so ``X_cam = R @ X_world + t``
- ``t`` and mesh vertices are in meters
- Neural Body's ``annots.npy`` stores the same ``t`` in millimeters

SAM 3D Body returns a mesh in the camera frame. Adding ``pred_cam_t`` to
``pred_vertices`` matches the official visualizer. Those camera-frame
points are then mapped into the shared calibration world.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Camera:
    camera_id: str
    view_name: str
    width: int
    height: int
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    distortion: np.ndarray

    def center_world(self) -> np.ndarray:
        """Camera center in world meters."""
        return -self.R.T @ self.t


@dataclass(frozen=True)
class BoxDetection:
    xyxy: np.ndarray
    confidence: float
    class_name: str


def _as_3x3(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {array.shape}")
    return array


def _as_vec3(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 3:
        raise ValueError(f"{name} must have 3 values, got {array.size}")
    return array


def load_cameras(path: str | Path) -> list[Camera]:
    """Load the shared-world camera file written by ``calibrate_stereo.py``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cameras = []
    for index, raw in enumerate(payload["cameras"]):
        distortion = np.asarray(raw.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64)
        distortion = distortion.reshape(-1)
        camera = Camera(
            camera_id=str(raw.get("camera_id", index)),
            view_name=str(raw.get("view_name", index)),
            width=int(raw["width"]),
            height=int(raw["height"]),
            K=_as_3x3(raw["K"], f"{raw.get('camera_id', index)}.K"),
            R=_as_3x3(raw["R"], f"{raw.get('camera_id', index)}.R"),
            t=_as_vec3(raw["t"], f"{raw.get('camera_id', index)}.t"),
            distortion=distortion,
        )
        if abs(np.linalg.det(camera.R) - 1.0) > 1e-2:
            raise ValueError(
                f"{camera.camera_id} rotation determinant is {np.linalg.det(camera.R):.4f}; "
                "expected a proper rotation near +1"
            )
        cameras.append(camera)
    if len(cameras) < 2:
        raise ValueError("the camera file needs the back view and the side view")
    return cameras


def select_pair(
    cameras: list[Camera],
    back_name: str = "back",
    side_name: str = "side",
) -> tuple[Camera, Camera]:
    by_name = {camera.view_name: camera for camera in cameras}
    if back_name in by_name and side_name in by_name:
        return by_name[back_name], by_name[side_name]
    return cameras[0], cameras[1]


def look_at_w2c(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """World-to-camera rotation and translation for a camera looking at ``target``."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=np.float64)
    forward = _normalize(target - eye)
    right = _normalize(np.cross(forward, up))
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ eye
    return rotation, translation


def project_points(camera: Camera, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points. Returns pixel coordinates and camera-frame depth."""
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    camera_points = (camera.R @ points.T).T + camera.t.reshape(1, 3)
    depth = camera_points[:, 2]
    projected = (camera.K @ camera_points.T).T
    pixels = projected[:, :2] / depth.reshape(-1, 1)
    return pixels, depth


def triangulate_point(
    camera_a: Camera,
    pixel_a: np.ndarray,
    camera_b: Camera,
    pixel_b: np.ndarray,
) -> np.ndarray:
    """Linear triangulation of one pixel correspondence. Pixels are undistorted."""
    projection_a = _projection(camera_a)
    projection_b = _projection(camera_b)
    rows = np.vstack(
        [
            _dlt_rows(projection_a, pixel_a),
            _dlt_rows(projection_b, pixel_b),
        ]
    )
    _, _, vh = np.linalg.svd(rows)
    homogeneous = vh[-1]
    return homogeneous[:3] / homogeneous[3]


def optical_axis_angle_deg(camera_a: Camera, camera_b: Camera) -> float:
    """Angle between the two optical axes. A back/side pair is near 90 degrees."""
    forward_a = camera_a.R[2]
    forward_b = camera_b.R[2]
    cosine = float(np.clip(np.dot(forward_a, forward_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def baseline_meters(camera_a: Camera, camera_b: Camera) -> float:
    return float(np.linalg.norm(camera_a.center_world() - camera_b.center_world()))


def undistort_pixel(camera: Camera, pixel: np.ndarray) -> np.ndarray:
    """Apply OpenCV undistortion when the camera has distortion coefficients."""
    coefficients = np.asarray(camera.distortion, dtype=np.float64).reshape(-1)
    if coefficients.size == 0 or np.linalg.norm(coefficients) < 1e-12:
        return np.asarray(pixel, dtype=np.float64).reshape(2)
    import cv2

    source = np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2)
    undistorted = cv2.undistortPoints(source, camera.K, coefficients, P=camera.K)
    return undistorted.reshape(2)


class FrameUndistorter:
    """Undistort frames once, then run detection and mesh recovery on the result.

    Neural Body also undistorts when ``D`` is nonzero. Exporting a zero
    distortion vector with the updated ``K`` keeps that second pass from
    moving the pixels again.
    """

    def __init__(self, camera: Camera):
        self.source = camera
        coefficients = np.asarray(camera.distortion, dtype=np.float64).reshape(-1)
        self.enabled = coefficients.size > 0 and float(np.linalg.norm(coefficients)) > 1e-12
        if self.enabled:
            import cv2

            new_k, _ = cv2.getOptimalNewCameraMatrix(
                camera.K,
                coefficients,
                (camera.width, camera.height),
                0,
            )
            self.camera = Camera(
                camera_id=camera.camera_id,
                view_name=camera.view_name,
                width=camera.width,
                height=camera.height,
                K=np.asarray(new_k, dtype=np.float64),
                R=camera.R.copy(),
                t=camera.t.copy(),
                distortion=np.zeros(5, dtype=np.float64),
            )
            self._coefficients = coefficients
        else:
            self.camera = camera

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return frame_bgr
        import cv2

        return cv2.undistort(
            frame_bgr,
            self.source.K,
            self._coefficients,
            None,
            self.camera.K,
        )


def ensure_frame_size(frame_bgr: np.ndarray, camera: Camera) -> None:
    height, width = frame_bgr.shape[:2]
    if width != camera.width or height != camera.height:
        raise ValueError(
            f"{camera.camera_id} frame is {width}x{height}, "
            f"but the camera file says {camera.width}x{camera.height}. "
            "Intrinsics belong to the file's resolution."
        )


def box_mask(shape: tuple[int, int], xyxy: np.ndarray) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = [int(round(float(value))) for value in xyxy]
    x1 = min(max(x1, 0), width - 1)
    x2 = min(max(x2, 0), width - 1)
    y1 = min(max(y1, 0), height - 1)
    y2 = min(max(y2, 0), height - 1)
    if x2 < x1 or y2 < y1:
        return np.zeros(shape, dtype=bool)
    mask = np.zeros(shape, dtype=bool)
    mask[y1 : y2 + 1, x1 : x2 + 1] = True
    return mask


def pad_box(xyxy: np.ndarray, padding: int, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    return np.array(
        [
            max(0.0, x1 - padding),
            max(0.0, y1 - padding),
            min(width - 1.0, x2 + padding),
            min(height - 1.0, y2 + padding),
        ],
        dtype=np.float64,
    )


def dilate_disk(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    try:
        import cv2
    except ImportError:
        return _dilate_disk_numpy(mask, radius)
    kernel_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def separate_player_and_ball(
    player_mask: np.ndarray,
    ball_mask: np.ndarray | None,
    dilation_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the ball from the body mask.

    The saved ball mask stays at the segmented size. A dilated copy is
    subtracted from the player so ball pixels, including the soft edge,
    never supervise the body radiance field. The composited sphere covers
    that gap. Fingers that overlap the dilated ball are absent from the
    body for that frame; that is preferable to an orange patch on the hand.
    """
    player = player_mask.astype(bool)
    if ball_mask is None or not np.any(ball_mask):
        return player, np.zeros(player.shape, dtype=bool)
    ball = ball_mask.astype(bool)
    halo = dilate_disk(ball, dilation_px)
    return player & ~halo, ball


def mask_centroid(mask: np.ndarray, fallback_box: np.ndarray | None) -> np.ndarray | None:
    if mask is not None and np.any(mask):
        ys, xs = np.nonzero(mask)
        return np.array([xs.mean(), ys.mean()], dtype=np.float64)
    if fallback_box is None:
        return None
    x1, y1, x2, y2 = [float(value) for value in fallback_box]
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)


def basketball_radius_m(
    camera: Camera,
    box: np.ndarray,
    depth_m: float,
) -> float:
    """Radius from the box's short side and the triangulated depth."""
    short_side = min(float(box[2] - box[0]), float(box[3] - box[1]))
    radius_px = 0.5 * max(short_side, 1.0)
    focal = float(camera.K[0, 0])
    return radius_px * depth_m / focal


def vertices_in_camera(person: dict, translation_mode: str) -> np.ndarray:
    vertices = np.asarray(person["pred_vertices"], dtype=np.float64)
    if translation_mode == "vertices_are_camera_frame":
        return vertices
    if translation_mode != "add_pred_cam_t":
        raise ValueError(f"unknown mesh translation mode: {translation_mode}")
    translation = np.asarray(person.get("pred_cam_t", np.zeros(3)), dtype=np.float64).reshape(-1)
    if translation.size != 3:
        raise ValueError("pred_cam_t must have 3 values")
    return vertices + translation.reshape(1, 3)


def camera_vertices_to_world(vertices_camera: np.ndarray, camera: Camera) -> np.ndarray:
    points = np.asarray(vertices_camera, dtype=np.float64).reshape(-1, 3)
    return (camera.R.T @ (points - camera.t.reshape(1, 3)).T).T


def mesh_confidence(person: dict, image_hw: tuple[int, int]) -> float:
    keypoints = np.asarray(person.get("pred_keypoints_2d", []), dtype=np.float64)
    if keypoints.ndim != 2 or keypoints.shape[0] == 0:
        return 0.01
    if keypoints.shape[1] >= 3:
        confidence = keypoints[:, 2]
        visible = confidence > 0.05
        if not np.any(visible):
            return 0.01
        return float(np.clip(confidence[visible].mean(), 0.01, 1.0))
    height, width = image_hw
    xy = keypoints[:, :2]
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    return float(max(inside.mean(), 0.01))


def fuse_world_meshes(
    world_vertices: list[np.ndarray],
    confidences: list[float],
    agreement_m: float,
) -> tuple[np.ndarray, str, float]:
    """Fuse same-topology world meshes.

    Vertex averaging is used only when the two posed meshes already agree.
    Averaging MHR joint angles is a different and worse operation: the
    angle vectors do not live in the camera frame, and a bad hand estimate
    from one view would bend the other view's hand. A large disagreement
    keeps the more confident view intact instead of collapsing the limb
    to the midpoint.
    """
    if not world_vertices:
        raise ValueError("no meshes to fuse")
    if len(world_vertices) == 1:
        return world_vertices[0], "single_view", float("nan")
    first, second = world_vertices[0], world_vertices[1]
    if first.shape != second.shape:
        winner = int(np.argmax(confidences))
        return world_vertices[winner], "best_view", float("nan")
    rmse = float(np.sqrt(np.mean((first - second) ** 2)))
    if rmse > agreement_m:
        winner = int(np.argmax(confidences))
        return world_vertices[winner], "best_view", rmse
    weights = np.asarray(confidences, dtype=np.float64)
    weights = weights / weights.sum()
    fused = weights[0] * first + weights[1] * second
    return fused, "average", rmse


def ball_disk_mask(
    camera: Camera,
    center_world: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    """Boolean image disk of a sphere. Empty when the ball is behind the camera."""
    mask = np.zeros((camera.height, camera.width), dtype=bool)
    pixels, depth = project_points(camera, center_world)
    z = float(depth[0])
    if not np.isfinite(z) or z <= 0.05 or not np.isfinite(radius_m) or radius_m <= 0:
        return mask
    u, v = pixels[0]
    radius_px = radius_m * float(camera.K[0, 0]) / z
    if radius_px < 0.5:
        return mask
    yy, xx = np.ogrid[: camera.height, : camera.width]
    mask = (xx - u) ** 2 + (yy - v) ** 2 <= radius_px**2
    return mask


def _projection(camera: Camera) -> np.ndarray:
    extrinsic = np.concatenate([camera.R, camera.t.reshape(3, 1)], axis=1)
    return camera.K @ extrinsic


def _dlt_rows(projection: np.ndarray, pixel: np.ndarray) -> np.ndarray:
    u, v = float(pixel[0]), float(pixel[1])
    return np.stack(
        [
            u * projection[2] - projection[0],
            v * projection[2] - projection[1],
        ]
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        raise ValueError("cannot normalize a near-zero vector")
    return vector / length


def _dilate_disk_numpy(mask: np.ndarray, radius: int) -> np.ndarray:
    source = mask.astype(bool)
    height, width = source.shape
    dilated = np.zeros_like(source)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            y_src_0 = max(0, -dy)
            y_src_1 = min(height, height - dy)
            x_src_0 = max(0, -dx)
            x_src_1 = min(width, width - dx)
            dilated[y_src_0 + dy : y_src_1 + dy, x_src_0 + dx : x_src_1 + dx] |= source[
                y_src_0:y_src_1, x_src_0:x_src_1
            ]
    return dilated
