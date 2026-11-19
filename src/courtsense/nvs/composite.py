"""Paint the triangulated basketball onto a rendered novel view."""

from __future__ import annotations

import numpy as np

from courtsense.nvs.geometry import Camera, ball_disk_mask, project_points


def composite_ball(
    image_bgr: np.ndarray,
    camera: Camera,
    center_world: np.ndarray,
    radius_m: float,
    depth_m: np.ndarray | None = None,
) -> np.ndarray:
    """Draw the ball as its own object.

    ``depth_m`` is optional camera-frame depth from the body renderer.
    Non-positive depth is treated as empty space. A surface closer than the
    front of the sphere hides the ball, so the sphere does not paint over
    the body when the player is in front.
    """
    mask = ball_disk_mask(camera, center_world, radius_m)
    if depth_m is not None and mask.any():
        _, depth = project_points(camera, center_world)
        front = float(depth[0]) - float(radius_m)
        surface = np.asarray(depth_m, dtype=np.float64)
        occluded = (surface > 0) & (surface < front)
        mask = mask & ~occluded
    if not mask.any():
        return image_bgr
    output = image_bgr.copy()
    pixels, depth = project_points(camera, center_world)
    u, v = pixels[0]
    radius_px = float(radius_m) * float(camera.K[0, 0]) / float(depth[0])
    yy, xx = np.ogrid[: camera.height, : camera.width]
    distance = np.sqrt((xx - u) ** 2 + (yy - v) ** 2)
    shade = np.clip(1.0 - 0.45 * (distance / max(radius_px, 1.0)), 0.55, 1.0)
    orange = np.array([30, 110, 220], dtype=np.float64)
    painted = orange.reshape(1, 1, 3) * shade[:, :, None]
    output[mask] = painted[mask].astype(np.uint8)
    return output
