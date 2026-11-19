"""Write a Neural Body / ZJU-MoCap directory plus the MHR and ball sidecars.

Neural Body reads ``annots.npy``, ``images/{camera}/{frame}.jpg`` and
``mask/{camera}/{frame}.png``. Masks in ``mask/`` are the player with the
ball removed. The ball has its own mask directory and per-frame track, which
the body model never reads.

``annots["cams"]["T"]`` is in millimeters, matching ZJU-MoCap. Vertices in
``vertices/{frame}.npy`` are world-frame meters.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from courtsense.nvs.geometry import Camera


def build_annots(cameras: list[Camera], frame_count: int) -> dict:
    return {
        "cams": {
            "K": [camera.K.tolist() for camera in cameras],
            "R": [camera.R.tolist() for camera in cameras],
            "T": [(camera.t.reshape(3, 1) * 1000.0).tolist() for camera in cameras],
            "D": [_distortion_column(camera) for camera in cameras],
        },
        "ims": [
            {"ims": [f"images/{camera_index}/{frame_index:06d}.jpg" for camera_index in range(len(cameras))]}
            for frame_index in range(frame_count)
        ],
    }


def save_annots(path: Path, cameras: list[Camera], frame_count: int) -> None:
    container = np.empty((), dtype=object)
    container[()] = build_annots(cameras, frame_count)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, container, allow_pickle=True)


def camera_to_json(camera: Camera) -> dict:
    return {
        "camera_id": camera.camera_id,
        "view_name": camera.view_name,
        "width": camera.width,
        "height": camera.height,
        "K": camera.K.tolist(),
        "R": camera.R.tolist(),
        "t": camera.t.tolist(),
        "distortion": np.asarray(camera.distortion, dtype=np.float64).reshape(-1).tolist(),
    }


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray | None) -> None:
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    triangles = None if faces is None else np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write("comment CourtSense MHR mesh, world meters\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        if triangles is not None:
            handle.write(f"element face {len(triangles)}\n")
            handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for x, y, z in points:
            handle.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        if triangles is not None:
            for a, b, c in triangles:
                handle.write(f"3 {a} {b} {c}\n")


@dataclass
class FrameRecord:
    export_index: int
    source_frame_index: int
    fusion_mode: str
    mesh_rmse_m: float | None
    view_confidence: dict[str, float]
    ball_valid: bool
    ball_reprojection_px: float | None
    radius_m: float | None
    radius_clamped: bool
    mask_source: dict[str, str]


@dataclass
class DatasetWriter:
    output_dir: Path
    cameras: list[Camera]
    overwrite: bool = False
    resume: bool = False
    records: list[FrameRecord] = field(default_factory=list)
    ball_centers: list[np.ndarray] = field(default_factory=list)
    ball_radii: list[float] = field(default_factory=list)
    ball_raw_radii: list[float] = field(default_factory=list)
    ball_clamped: list[bool] = field(default_factory=list)
    ball_valid: list[bool] = field(default_factory=list)
    ball_reprojection: list[float] = field(default_factory=list)
    _faces_written: bool = False

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.resume:
            self._load_existing()
            return
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            if not self.overwrite:
                raise FileExistsError(
                    f"{self.output_dir} is not empty. Pass --overwrite to replace the export."
                )
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_camera_file()

    def write_views(
        self,
        record: FrameRecord,
        images_bgr: list[np.ndarray],
        player_masks: list[np.ndarray],
        ball_masks: list[np.ndarray],
        detections: list[dict],
        ball: dict,
    ) -> None:
        index = record.export_index
        for camera_index, image in enumerate(images_bgr):
            image_path = self.output_dir / "images" / str(camera_index) / f"{index:06d}.jpg"
            _write_image(image_path, image)
            _write_mask(
                self.output_dir / "mask" / str(camera_index) / f"{index:06d}.png",
                player_masks[camera_index],
            )
            _write_mask(
                self.output_dir / "mask_ball" / str(camera_index) / f"{index:06d}.png",
                ball_masks[camera_index],
            )
            detection_path = self.output_dir / "detections" / str(camera_index) / f"{index:06d}.json"
            detection_path.parent.mkdir(parents=True, exist_ok=True)
            detection_path.write_text(json.dumps(detections[camera_index]), encoding="utf-8")
        ball_path = self.output_dir / "ball" / f"{index:06d}.npz"
        ball_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ball_path, **ball)
        self.records.append(record)
        self.ball_centers.append(np.asarray(ball["center_world"], dtype=np.float64))
        self.ball_radii.append(float(ball["radius_m"]))
        self.ball_raw_radii.append(float(ball["radius_raw_m"]))
        self.ball_clamped.append(bool(ball["radius_clamped"]))
        self.ball_valid.append(bool(ball["valid"]))
        self.ball_reprojection.append(float(ball["reprojection_px"]))
        self._append_record(record)
        self._write_trajectory()

    def write_meshes(
        self,
        export_index: int,
        fused_vertices: np.ndarray | None,
        per_view: list[dict | None],
        faces: np.ndarray | None,
        fusion_mode: str,
        mesh_rmse_m: float | None,
        view_confidence: dict[str, float],
    ) -> None:
        if fused_vertices is not None:
            vertex_path = self.output_dir / "vertices" / f"{export_index:06d}.npy"
            vertex_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(vertex_path, np.asarray(fused_vertices, dtype=np.float32))
            if export_index == 0 or not self._faces_written:
                write_ply(self.output_dir / "vertices" / f"{export_index:06d}.ply", fused_vertices, faces)
        if faces is not None and not self._faces_written:
            face_path = self.output_dir / "mhr_faces.npy"
            np.save(face_path, np.asarray(faces, dtype=np.int64))
            self._faces_written = True
        for camera_index, payload in enumerate(per_view):
            if payload is None:
                continue
            path = self.output_dir / "mhr" / str(camera_index) / f"{export_index:06d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **payload)
        self._update_mesh_record(export_index, fusion_mode, mesh_rmse_m, view_confidence)

    def close(self, extra_metadata: dict | None = None) -> dict:
        frame_count = len(self.records)
        save_annots(self.output_dir / "annots.npy", self.cameras, frame_count)
        summary = self.summary()
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        metadata = {
            "format": "neuralbody-zju-mocap-v1+mhr",
            "nerf_repository": "https://github.com/zju3dv/neuralbody",
            "vertex_units": "meters",
            "vertex_frame": "world",
            "camera_convention": "opencv_world_to_camera",
            "annots_translation_units": "millimeters",
            "human_mask": "mask/: player with the ball removed",
            "ball_mask": "mask_ball/: ball only, composited after the body render",
            "images_undistorted": True,
            "frame_count": frame_count,
            "cameras": [camera_to_json(camera) for camera in self.cameras],
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        (self.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._write_trajectory()
        return summary

    def summary(self) -> dict:
        modes = [record.fusion_mode for record in self.records]
        rmse_values = [
            record.mesh_rmse_m
            for record in self.records
            if record.mesh_rmse_m is not None and np.isfinite(record.mesh_rmse_m)
        ]
        reprojection = [
            record.ball_reprojection_px
            for record in self.records
            if record.ball_valid and record.ball_reprojection_px is not None
        ]
        return {
            "frames_exported": len(self.records),
            "fusion_average": modes.count("average"),
            "fusion_best_view": modes.count("best_view"),
            "fusion_single_view": modes.count("single_view"),
            "fusion_skipped": modes.count("skipped"),
            "ball_triangulated": int(sum(self.ball_valid)),
            "ball_missing": int(len(self.ball_valid) - sum(self.ball_valid)),
            "mean_mesh_rmse_m": float(np.mean(rmse_values)) if rmse_values else None,
            "mean_ball_reprojection_px": float(np.mean(reprojection)) if reprojection else None,
        }

    def _load_existing(self) -> None:
        record_path = self.output_dir / "frames.jsonl"
        if not record_path.exists():
            raise FileNotFoundError(f"{record_path} is missing; run the detection pass first")
        self.records = []
        for line in record_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            self.records.append(
                FrameRecord(
                    export_index=int(raw["export_index"]),
                    source_frame_index=int(raw["source_frame_index"]),
                    fusion_mode=str(raw["fusion_mode"]),
                    mesh_rmse_m=raw["mesh_rmse_m"],
                    view_confidence=dict(raw["view_confidence"]),
                    ball_valid=bool(raw["ball_valid"]),
                    ball_reprojection_px=raw["ball_reprojection_px"],
                    radius_m=raw["radius_m"],
                    radius_clamped=bool(raw["radius_clamped"]),
                    mask_source=dict(raw["mask_source"]),
                )
            )
        trajectory = self.output_dir / "ball_trajectory.npz"
        if trajectory.exists():
            track = np.load(trajectory)
            self.ball_centers = list(track["center_world"])
            self.ball_radii = list(track["radius_m"])
            self.ball_raw_radii = list(track["radius_raw_m"])
            self.ball_clamped = list(track["radius_clamped"])
            self.ball_valid = list(track["valid"])
            self.ball_reprojection = list(track["reprojection_px"])
        self._faces_written = (self.output_dir / "mhr_faces.npy").exists()

    def _write_camera_file(self) -> None:
        payload = {
            "note": "Cameras actually used for the export. Images are undistorted and distortion is zero.",
            "world_units": "meters",
            "opencv_convention": "x_right_y_down_z_forward_world_to_camera",
            "cameras": [camera_to_json(camera) for camera in self.cameras],
        }
        (self.output_dir / "cameras_export.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _append_record(self, record: FrameRecord) -> None:
        path = self.output_dir / "frames.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_record_dict(record)) + "\n")

    def _update_mesh_record(
        self,
        export_index: int,
        fusion_mode: str,
        mesh_rmse_m: float | None,
        view_confidence: dict[str, float],
    ) -> None:
        for record in self.records:
            if record.export_index == export_index:
                record.fusion_mode = fusion_mode
                record.mesh_rmse_m = mesh_rmse_m
                record.view_confidence = view_confidence
        path = self.output_dir / "frames.jsonl"
        lines = [
            json.dumps(_record_dict(record))
            for record in sorted(self.records, key=lambda item: item.export_index)
        ]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _write_trajectory(self) -> None:
        if not self.ball_centers:
            return
        path = self.output_dir / "ball_trajectory.npz"
        np.savez_compressed(
            path,
            center_world=np.stack(self.ball_centers, axis=0),
            radius_m=np.asarray(self.ball_radii, dtype=np.float64),
            radius_raw_m=np.asarray(self.ball_raw_radii, dtype=np.float64),
            radius_clamped=np.asarray(self.ball_clamped, dtype=bool),
            valid=np.asarray(self.ball_valid, dtype=bool),
            reprojection_px=np.asarray(self.ball_reprojection, dtype=np.float64),
        )


def _record_dict(record: FrameRecord) -> dict:
    return {
        "export_index": record.export_index,
        "source_frame_index": record.source_frame_index,
        "fusion_mode": record.fusion_mode,
        "mesh_rmse_m": record.mesh_rmse_m,
        "view_confidence": record.view_confidence,
        "ball_valid": record.ball_valid,
        "ball_reprojection_px": record.ball_reprojection_px,
        "radius_m": record.radius_m,
        "radius_clamped": record.radius_clamped,
        "mask_source": record.mask_source,
    }


def _distortion_column(camera: Camera) -> list:
    distortion = np.asarray(camera.distortion, dtype=np.float64).reshape(-1)
    if distortion.size == 0:
        distortion = np.zeros(5, dtype=np.float64)
    return distortion.reshape(-1, 1).tolist()


def _write_image(path: Path, image_bgr: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise RuntimeError(f"failed to write {path}")


def _write_mask(path: Path, mask: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.where(mask.astype(bool), 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write {path}")
