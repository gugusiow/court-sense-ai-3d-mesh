"""Checkerboard calibration into one shared world frame.

Each camera is calibrated on its own. The shared frame is the checkerboard
on the simultaneous view with the lowest triangulation error. Lay the board
flat on the court if the exported Z axis should point up.
"""

from __future__ import annotations

import numpy as np


def checkerboard_object_points(cols: int, rows: int, square_m: float) -> np.ndarray:
    """Inner-corner coordinates on Z=0, origin at the first detected corner."""
    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack(
        [
            grid_x.reshape(-1) * square_m,
            grid_y.reshape(-1) * square_m,
            np.zeros(cols * rows),
        ],
        axis=1,
    )
    return points.astype(np.float64)


def pose_from_corners(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """solvePnP pose and mean reprojection error in pixels."""
    import cv2

    image = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    object_shaped = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    ok, rvec, tvec = cv2.solvePnP(
        object_shaped,
        image,
        np.asarray(K, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64).reshape(-1),
    )
    if not ok:
        raise RuntimeError("solvePnP failed")
    rotation, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(
        object_shaped,
        rvec,
        tvec,
        np.asarray(K, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64).reshape(-1),
    )
    error = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - image.reshape(-1, 2), axis=1)))
    return rotation, tvec.reshape(3), error


def calibrate_intrinsics(
    object_points: np.ndarray,
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    import cv2

    # OpenCV 5 calibrateCamera accepts Point3f / Point2f only.
    objects = [np.asarray(object_points, dtype=np.float32).reshape(-1, 1, 3) for _ in image_points]
    images = [np.asarray(points, dtype=np.float32).reshape(-1, 1, 2) for points in image_points]
    rms, K, distortion, _, _ = cv2.calibrateCamera(objects, images, image_size, None, None)
    return np.asarray(K, dtype=np.float64), np.asarray(distortion, dtype=np.float64).reshape(-1), float(rms)


def select_shared_board_pose(
    object_points: np.ndarray,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    back_K: np.ndarray,
    back_distortion: np.ndarray,
    side_K: np.ndarray,
    side_distortion: np.ndarray,
    back_size: tuple[int, int],
    side_size: tuple[int, int],
) -> dict:
    """Pick the corner order and frame whose two poses triangulate back to the board."""
    from courtsense.nvs.geometry import Camera, triangulate_point, undistort_pixel

    best: dict | None = None
    for pair_index, (back_corners, side_corners) in enumerate(pairs):
        for order_name, side_image in (
            ("as_detected", side_corners),
            ("reversed", side_corners[::-1].copy()),
        ):
            rotation_back, translation_back, error_back = pose_from_corners(
                object_points, back_corners, back_K, back_distortion
            )
            rotation_side, translation_side, error_side = pose_from_corners(
                object_points, side_image, side_K, side_distortion
            )
            back_camera = _board_camera(
                "back", back_size, back_K, rotation_back, translation_back, back_distortion
            )
            side_camera = _board_camera(
                "side", side_size, side_K, rotation_side, translation_side, side_distortion
            )
            residuals = []
            for point, back_uv, side_uv in zip(object_points, back_corners.reshape(-1, 2), side_image.reshape(-1, 2)):
                recovered = triangulate_point(
                    back_camera,
                    undistort_pixel(back_camera, back_uv),
                    side_camera,
                    undistort_pixel(side_camera, side_uv),
                )
                residuals.append(float(np.linalg.norm(recovered - point)))
            alignment = float(np.mean(residuals))
            candidate = {
                "pair_index": pair_index,
                "corner_order": order_name,
                "alignment_rmse_m": alignment,
                "back_reprojection_px": error_back,
                "side_reprojection_px": error_side,
                "back_R": rotation_back,
                "back_t": translation_back,
                "side_R": rotation_side,
                "side_t": translation_side,
            }
            if best is None or alignment + 1e-4 < best["alignment_rmse_m"]:
                best = candidate
    if best is None:
        raise RuntimeError("no simultaneous checkerboard views were provided")
    return best


def _board_camera(view_name, image_size, K, rotation, translation, distortion) -> "Camera":
    from courtsense.nvs.geometry import Camera

    width, height = image_size
    return Camera(
        camera_id=view_name,
        view_name=view_name,
        width=width,
        height=height,
        K=np.asarray(K, dtype=np.float64),
        R=np.asarray(rotation, dtype=np.float64),
        t=np.asarray(translation, dtype=np.float64),
        distortion=np.asarray(distortion, dtype=np.float64).reshape(-1),
    )
