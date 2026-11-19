"""Deterministic stand-ins used by ``--self-check`` and the unit tests.

The meshes are the same world points expressed in each camera, so a correct
fusion reports ``average`` and a near-zero RMSE. The ball box is the
projection of a known world point, so triangulation has to recover it.
"""

from __future__ import annotations

import numpy as np

from courtsense.nvs.geometry import BoxDetection, Camera, box_mask, project_points


class SyntheticDetector:
    def __init__(self, camera: Camera, ball_world: np.ndarray, player_xyxy: np.ndarray):
        self.camera = camera
        self.ball_world = np.asarray(ball_world, dtype=np.float64)
        self.player_xyxy = np.asarray(player_xyxy, dtype=np.float64)

    def detect(self, frame_bgr: np.ndarray) -> tuple[BoxDetection, BoxDetection]:
        uv, depth = project_points(self.camera, self.ball_world)
        if float(depth[0]) <= 0:
            raise RuntimeError("synthetic ball is behind the camera")
        u, v = uv[0]
        half = 5.0
        ball_box = np.array([u - half, v - half, u + half, v + half], dtype=np.float64)
        return (
            BoxDetection(self.player_xyxy.copy(), 0.95, "person"),
            BoxDetection(ball_box, 0.9, "sports ball"),
        )


class SyntheticSegmenter:
    def segment(self, frame_bgr, player_box, ball_box):
        shape = frame_bgr.shape[:2]
        player = None if player_box is None else box_mask(shape, player_box)
        ball = None if ball_box is None else box_mask(shape, ball_box)
        return player, ball


class SyntheticMesher:
    def __init__(self, cameras: list[Camera], world_vertices: np.ndarray):
        self.cameras = list(cameras)
        self.world_vertices = np.asarray(world_vertices, dtype=np.float64)
        self.faces = np.array([[0, 1, 2]], dtype=np.int64)

    def estimate(self, frame_bgr, player_box, player_mask, camera: Camera):
        known = {item.camera_id for item in self.cameras}
        if camera.camera_id not in known:
            raise KeyError(f"synthetic scene has no camera {camera.camera_id}")
        camera_vertices = (camera.R @ self.world_vertices.T).T + camera.t.reshape(1, 3)
        keypoints = np.ones((8, 3), dtype=np.float64)
        return {
            "pred_vertices": camera_vertices,
            "pred_cam_t": np.zeros(3, dtype=np.float64),
            "pred_keypoints_2d": keypoints,
            "body_pose_params": np.zeros(4, dtype=np.float64),
            "hand_pose_params": np.zeros(4, dtype=np.float64),
            "shape_params": np.ones(3, dtype=np.float64),
            "mhr_model_params": np.zeros(8, dtype=np.float64),
        }


def make_practice_scene(width: int = 80, height: int = 60):
    """Back and side cameras looking at a small mesh and a ball."""
    from courtsense.nvs.geometry import look_at_w2c

    focal = 90.0
    intrinsic = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    target = np.array([0.0, 0.0, 1.2])
    cameras = []
    for camera_id, view_name, eye in (
        ("cam_back", "back", np.array([0.0, -4.0, 1.6])),
        ("cam_side", "side", np.array([4.0, 0.0, 1.6])),
    ):
        rotation, translation = look_at_w2c(eye, target)
        cameras.append(
            Camera(
                camera_id=camera_id,
                view_name=view_name,
                width=width,
                height=height,
                K=intrinsic.copy(),
                R=rotation,
                t=translation,
                distortion=np.zeros(5),
            )
        )
    world_vertices = target.reshape(1, 3) + np.array(
        [
            [0.0, 0.0, 0.0],
            [0.12, 0.0, 0.05],
            [-0.08, 0.04, 0.15],
            [0.0, -0.05, -0.2],
        ],
        dtype=np.float64,
    )
    ball_world = np.array([0.25, 0.1, 1.35], dtype=np.float64)
    player_box = np.array([8, 6, width - 8, height - 6], dtype=np.float64)
    return cameras, world_vertices, ball_world, player_box
