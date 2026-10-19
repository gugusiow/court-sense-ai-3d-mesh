"""Atomic local artifact storage."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from courtsense.domain.models import FeatureRecord, QualityReport, ReconstructionSequence
from courtsense.persistence.database import MetadataStore


class ArtifactStore:
    def __init__(self, root: str | Path, metadata: MetadataStore):
        self.root = Path(root)
        self.metadata = metadata
        self.root.mkdir(parents=True, exist_ok=True)

    def _finalize(self, artifact_id, artifact_type, owner_id, suffix, writer, processing_version):
        relative = Path(artifact_type) / f"{artifact_id}{suffix}"
        final_path = self.root / relative
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            return artifact_id
        fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=final_path.parent)
        os.close(fd)
        temporary_path = Path(temporary)
        try:
            writer(temporary_path)
            checksum = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
            os.replace(temporary_path, final_path)
            self.metadata.add_artifact(
                artifact_id, artifact_type, owner_id, str(relative), checksum, processing_version
            )
        finally:
            temporary_path.unlink(missing_ok=True)
        return artifact_id

    def write_sequence(self, sequence: ReconstructionSequence, owner_id: str) -> str:
        artifact_id = f"{owner_id}:trajectory:{sequence.reconstruction_version}"

        def writer(path: Path) -> None:
            with path.open("wb") as handle:
                np.savez_compressed(
                    handle, timestamps_ns=sequence.timestamps_ns, joints=sequence.joints,
                    validity_mask=sequence.validity_mask, joint_names=np.array(sequence.joint_names),
                    coordinate_system=sequence.coordinate_system, units=sequence.units,
                    provenance=sequence.provenance.value,
                )

        return self._finalize(
            artifact_id, "trajectories", owner_id, ".npz", writer, sequence.reconstruction_version
        )

    def write_quality(self, report: QualityReport, owner_id: str) -> str:
        artifact_id = f"{owner_id}:quality:{report.quality_version}"

        def writer(path: Path) -> None:
            path.write_text(json.dumps({
                "sequence_id": report.sequence_id, "status": report.status.value,
                "quality_version": report.quality_version,
                "checks": [check.__dict__ for check in report.checks],
            }, indent=2))

        return self._finalize(
            artifact_id, "quality", owner_id, ".json", writer, report.quality_version
        )

    def write_features(
        self, features: tuple[FeatureRecord, ...], owner_id: str, processing_version: str
    ) -> str:
        import pandas as pd

        artifact_id = f"{owner_id}:features:{processing_version}"

        def writer(path: Path) -> None:
            pd.DataFrame([feature.__dict__ for feature in features]).to_parquet(path, index=False)

        return self._finalize(
            artifact_id, "features", owner_id, ".parquet", writer, processing_version
        )

    def path_for(self, artifact_id: str) -> Path:
        matches = list(self.root.rglob(f"{artifact_id}.*"))
        if len(matches) != 1:
            raise FileNotFoundError(f"artifact not found or ambiguous: {artifact_id}")
        return matches[0]

    def delete_owner(self, owner_id: str) -> None:
        for path in self.root.rglob("*"):
            if path.is_file() and owner_id in path.name:
                path.unlink()
