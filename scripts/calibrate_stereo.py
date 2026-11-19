"""Estimate one shared-world camera file from a checkerboard shown to both views.

The board has to be visible in both videos on the same frame index. Lay it
flat on the court so Z points up. ``--square-mm`` is the size of one square,
not the size of the printed page; a wrong value scales the whole NeRF scene.

Example:
python scripts/calibrate_stereo.py --video-back data/raw/calib_back.mp4 --video-side data/raw/calib_side.mp4 --square-mm 40 --output config/cameras.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from courtsense.nvs.calibration import (  # noqa: E402
    calibrate_intrinsics,
    checkerboard_object_points,
    select_shared_board_pose,
)


def main() -> None:
    args = _parse_args()
    import cv2

    object_points = checkerboard_object_points(args.board_cols, args.board_rows, args.square_mm / 1000.0)
    back_views, side_views, pairs, back_size, side_size = _collect(args, cv2)
    if len(back_views) < args.min_views or len(side_views) < args.min_views:
        raise SystemExit(
            f"Need at least {args.min_views} board detections per camera, "
            f"found {len(back_views)} and {len(side_views)}."
        )
    if not pairs:
        raise SystemExit("The board was never visible in both cameras on the same frame.")
    print(f"board views: back={len(back_views)} side={len(side_views)} simultaneous={len(pairs)}")
    back_K, back_dist, back_rms = calibrate_intrinsics(object_points, back_views, back_size)
    side_K, side_dist, side_rms = calibrate_intrinsics(object_points, side_views, side_size)
    print(f"intrinsic reprojection rms: back={back_rms:.3f}px side={side_rms:.3f}px")
    chosen = select_shared_board_pose(
        object_points,
        pairs,
        back_K,
        back_dist,
        side_K,
        side_dist,
        back_size,
        side_size,
    )
    if chosen["alignment_rmse_m"] > args.max_alignment_m:
        raise SystemExit(
            "The two cameras do not agree on the board "
            f"({chosen['alignment_rmse_m'] * 100:.1f} cm RMSE). "
            "Refilm with a larger board, slower motion, and the board filling both views."
        )
    payload = {
        "note": (
            "World origin is the checkerboard on the selected frame. "
            "Z=0 is the board plane. A board laid flat on the court has Z up."
        ),
        "world_units": "meters",
        "opencv_convention": "x_right_y_down_z_forward_world_to_camera",
        "calibration": {
            "selected_pair_index": chosen["pair_index"],
            "corner_order": chosen["corner_order"],
            "board_alignment_rmse_m": chosen["alignment_rmse_m"],
            "back_intrinsic_rms_px": back_rms,
            "side_intrinsic_rms_px": side_rms,
            "back_pose_reprojection_px": chosen["back_reprojection_px"],
            "side_pose_reprojection_px": chosen["side_reprojection_px"],
            "square_mm": args.square_mm,
            "board_inner_corners": [args.board_cols, args.board_rows],
        },
        "cameras": [
            _camera_dict(
                "cam_back",
                "back",
                back_size,
                back_K,
                chosen["back_R"],
                chosen["back_t"],
                back_dist,
            ),
            _camera_dict(
                "cam_side",
                "side",
                side_size,
                side_K,
                chosen["side_R"],
                chosen["side_t"],
                side_dist,
            ),
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"wrote {output} alignment={chosen['alignment_rmse_m'] * 1000:.1f} mm "
        f"order={chosen['corner_order']}"
    )


def _collect(args, cv2):
    back_cap = _open(args.video_back, cv2)
    side_cap = _open(args.video_side, cv2)
    back_size = _size(back_cap, cv2)
    side_size = _size(side_cap, cv2)
    pattern = (args.board_cols, args.board_rows)
    back_views = []
    side_views = []
    pairs = []
    scanned = 0
    while args.max_frames is None or scanned < args.max_frames:
        for _ in range(max(args.frame_stride - 1, 0)):
            if not back_cap.grab() or not side_cap.grab():
                back_cap.release()
                side_cap.release()
                return back_views, side_views, pairs, back_size, side_size
        ok_back, back = back_cap.read()
        ok_side, side = side_cap.read()
        if not ok_back or not ok_side:
            break
        scanned += 1
        back_corners = _find_corners(back, pattern, cv2)
        side_corners = _find_corners(side, pattern, cv2)
        if back_corners is not None:
            back_views.append(back_corners)
        if side_corners is not None:
            side_views.append(side_corners)
        if back_corners is not None and side_corners is not None:
            pairs.append((back_corners, side_corners))
            if len(pairs) >= args.max_pairs:
                break
    back_cap.release()
    side_cap.release()
    return back_views, side_views, pairs, back_size, side_size


def _find_corners(frame, pattern, cv2):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    corners = None
    if hasattr(cv2, "findChessboardCornersSB"):
        found, detected = cv2.findChessboardCornersSB(gray, pattern, cv2.CALIB_CB_EXHAUSTIVE)
        if found:
            corners = detected
    if corners is None:
        found, detected = cv2.findChessboardCorners(
            gray,
            pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            return None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        corners = cv2.cornerSubPix(gray, detected, (5, 5), (-1, -1), criteria)
    return np.asarray(corners, dtype=np.float64).reshape(-1, 2)


def _camera_dict(camera_id, view_name, image_size, K, rotation, translation, distortion):
    width, height = image_size
    return {
        "camera_id": camera_id,
        "view_name": view_name,
        "width": int(width),
        "height": int(height),
        "K": np.asarray(K, dtype=np.float64).tolist(),
        "R": np.asarray(rotation, dtype=np.float64).tolist(),
        "t": np.asarray(translation, dtype=np.float64).reshape(3).tolist(),
        "distortion": np.asarray(distortion, dtype=np.float64).reshape(-1).tolist(),
    }


def _open(path, cv2):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    return capture


def _size(capture, cv2):
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise SystemExit("could not read the video resolution")
    return width, height


def _parse_args():
    parser = argparse.ArgumentParser(description="Calibrate two cameras from a shared checkerboard.")
    parser.add_argument("--video-back", required=True)
    parser.add_argument("--video-side", required=True)
    parser.add_argument("--output", default="config/cameras.json")
    parser.add_argument("--square-mm", required=True, type=float, help="Physical size of one checkerboard square.")
    parser.add_argument("--board-cols", type=int, default=9, help="Inner corners along the board's width.")
    parser.add_argument("--board-rows", type=int, default=6, help="Inner corners along the board's height.")
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--max-pairs", type=int, default=400)
    parser.add_argument("--min-views", type=int, default=8)
    parser.add_argument("--max-alignment-m", type=float, default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    main()
