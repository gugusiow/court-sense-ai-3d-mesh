"""Paint the exported basketball onto a rendered frame.

The body renderer never saw the ball. This draws the triangulated sphere
afterwards, optionally hidden where a depth map says the body is in front.

Example:
python scripts/composite_ball.py --render render.png --ball data/nvs/shot/ball/000010.npz --cameras data/nvs/shot/cameras_export.json --camera-id cam_back --output composited.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from courtsense.nvs.composite import composite_ball  # noqa: E402
from courtsense.nvs.geometry import load_cameras  # noqa: E402


def main() -> None:
    args = _parse_args()
    import cv2

    cameras = load_cameras(args.cameras)
    camera = next((item for item in cameras if item.camera_id == args.camera_id or item.view_name == args.camera_id), None)
    if camera is None:
        known = ", ".join(item.camera_id for item in cameras)
        raise SystemExit(f"camera {args.camera_id!r} is not in {args.cameras}. Known ids: {known}")
    image = cv2.imread(args.render, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read {args.render}")
    if image.shape[1] != camera.width or image.shape[0] != camera.height:
        raise SystemExit(
            f"render is {image.shape[1]}x{image.shape[0]} and camera {camera.camera_id} is {camera.width}x{camera.height}"
        )
    ball = np.load(args.ball)
    if not bool(ball["valid"]):
        raise SystemExit(f"{args.ball} has no accepted triangulation")
    depth = None if not args.depth else np.load(args.depth)
    painted = composite_ball(image, camera, ball["center_world"], float(ball["radius_m"]), depth)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), painted):
        raise SystemExit(f"failed to write {output}")
    print(f"wrote {output}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Composite the triangulated ball onto a body render.")
    parser.add_argument("--render", required=True, help="Body-only rendered BGR image.")
    parser.add_argument("--ball", required=True, help="Per-frame ball npz from preprocess.py.")
    parser.add_argument("--cameras", required=True, help="cameras_export.json, or a virtual camera in the same JSON schema.")
    parser.add_argument("--camera-id", required=True, help="camera_id or view_name matching the render.")
    parser.add_argument("--depth", default="", help="Optional float32 camera-frame depth, same resolution as the render.")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
