"""Lazy adapters for YOLO, SAM 2, and SAM 3D Body.

Importing this module does not import those packages. Each adapter imports
its own stack in ``__init__`` so the geometry tests stay light.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

from courtsense.nvs.geometry import BoxDetection, Camera


class YoloDetector:
    """One YOLO model bound to one camera stream.

    ``predict`` is used instead of ``track``. A shared tracker would see the
    back-view player and the side-view player as one object jumping across
    the court. Build one detector per camera; the weights are loaded twice
    so the two streams stay independent even if Ultralytics caches state.
    """

    def __init__(
        self,
        weights: str,
        person_confidence: float = 0.35,
        ball_confidence: float = 0.15,
        device: str = "cpu",
    ):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.person_confidence = person_confidence
        self.ball_confidence = ball_confidence
        self.device = device
        names = _class_names(self.model.names)
        self.person_ids = {index for index, name in names.items() if name == "person"}
        self.ball_ids = {index for index, name in names.items() if name in {"sports ball", "basketball"}}
        if not self.person_ids or not self.ball_ids:
            raise RuntimeError(f"YOLO names did not include person and sports ball: {names}")

    def detect(self, frame_bgr: np.ndarray) -> tuple[BoxDetection | None, BoxDetection | None]:
        results = self.model.predict(
            frame_bgr,
            verbose=False,
            conf=min(self.person_confidence, self.ball_confidence),
            device=self.device,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None, None
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        confidence = boxes.conf.detach().cpu().numpy()
        xyxy = boxes.xyxy.detach().cpu().numpy()
        player = _best_person(xyxy, confidence, classes, self.person_ids, self.person_confidence)
        ball = _best_ball(xyxy, confidence, classes, self.ball_ids, self.ball_confidence)
        return player, ball


class Sam2Segmenter:
    """Per-frame SAM 2 image prompts.

    A video predictor prompted once on frame 0 smears the ball into the hand
    after the release. Prompting every frame with that frame's YOLO box keeps
    the ball mask on the ball when it leaves the body.
    """

    def __init__(self, config_file: str, checkpoint: str, device: str, repo_root: str | None = None):
        _prepend_repo(repo_root)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(config_file, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(model)

    def segment(
        self,
        frame_bgr: np.ndarray,
        player_box: np.ndarray | None,
        ball_box: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        import cv2

        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self.predictor.set_image(rgb)
        player_mask = _predict_box(self.predictor, player_box)
        ball_mask = _predict_box(self.predictor, ball_box)
        return player_mask, ball_mask


class Sam3DBodyEstimator:
    """Full-frame SAM 3D Body inference prompted by the player box and mask.

    The official estimator moves the batch with ``.cuda()``, so this adapter
    requires an NVIDIA GPU. ``inference_type="full"`` runs the hand decoder,
    which is the wrist and finger detail MHR is being used for.
    """

    def __init__(
        self,
        checkpoint: str,
        mhr_path: str,
        repo_root: str | None = None,
        inference_type: str = "full",
    ):
        _prepend_repo(repo_root)
        import torch
        from sam_3d_body import SAM3DBodyEstimator as Estimator
        from sam_3d_body import load_sam_3d_body

        if not torch.cuda.is_available():
            raise RuntimeError(
                "SAM 3D Body's released estimator calls .cuda() internally. "
                "Mesh inference needs an NVIDIA GPU. Re-run with --skip-mesh "
                "to export masks and the ball track first."
            )
        self.torch = torch
        self.inference_type = inference_type
        device = torch.device("cuda")
        model, model_cfg = load_sam_3d_body(checkpoint, device=device, mhr_path=mhr_path)
        # No internal detector or SAM 2. YOLO and the external SAM 2 masks are the prompts.
        self.estimator = Estimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
        )
        self.faces = np.asarray(self.estimator.faces, dtype=np.int64)
        self.device = device

    def estimate(
        self,
        frame_bgr: np.ndarray,
        player_box: np.ndarray,
        player_mask: np.ndarray,
        camera: Camera,
    ) -> dict | None:
        import cv2

        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        boxes = np.asarray(player_box, dtype=np.float32).reshape(1, 4)
        height, width = rgb.shape[:2]
        masks = None
        if player_mask is not None and np.any(player_mask):
            binary = player_mask.astype(np.uint8)
            masks = binary.reshape(1, height, width, 1)
        intrinsics = self.torch.as_tensor(camera.K, dtype=self.torch.float32, device=self.device).reshape(1, 3, 3)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            outputs = self.estimator.process_one_image(
                rgb,
                bboxes=boxes,
                masks=masks,
                cam_int=intrinsics,
                inference_type=self.inference_type,
            )
        if not outputs:
            return None
        person = outputs[0]
        return {key: value for key, value in person.items() if key != "mask"}


def _best_person(xyxy, confidence, classes, person_ids, threshold) -> BoxDetection | None:
    candidates = []
    for box, score, class_id in zip(xyxy, confidence, classes):
        if int(class_id) not in person_ids or float(score) < threshold:
            continue
        area = max(float(box[2] - box[0]), 0.0) * max(float(box[3] - box[1]), 0.0)
        candidates.append((area, float(score), box))
    if not candidates:
        return None
    _, score, box = max(candidates, key=lambda item: item[0])
    return BoxDetection(xyxy=np.asarray(box, dtype=np.float64), confidence=score, class_name="person")


def _best_ball(xyxy, confidence, classes, ball_ids, threshold) -> BoxDetection | None:
    candidates = []
    for box, score, class_id in zip(xyxy, confidence, classes):
        if int(class_id) not in ball_ids or float(score) < threshold:
            continue
        candidates.append((float(score), box))
    if not candidates:
        return None
    score, box = max(candidates, key=lambda item: item[0])
    return BoxDetection(xyxy=np.asarray(box, dtype=np.float64), confidence=score, class_name="sports ball")


def _predict_box(predictor, box: np.ndarray | None) -> np.ndarray | None:
    if box is None:
        return None
    masks, _, _ = predictor.predict(
        box=np.asarray(box, dtype=np.float32).reshape(1, 4),
        multimask_output=False,
    )
    mask = np.asarray(masks)
    return mask.reshape(mask.shape[-2], mask.shape[-1]) > 0


def _prepend_repo(repo_root: str | None) -> None:
    if not repo_root:
        return
    root = Path(repo_root).resolve()
    if not root.exists():
        return
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _class_names(names) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    return {index: str(name) for index, name in enumerate(names)}
