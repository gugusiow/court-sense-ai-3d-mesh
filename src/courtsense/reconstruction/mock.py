"""Synthetic reconstruction for development and automated tests."""

import numpy as np

from courtsense.domain.models import (
    Provenance,
    ReconstructionSequence,
    SynchronizedFramePair,
)


class MockReconstructionAdapter:
    """Produce a deterministic 3-joint trajectory, not a validated body model."""

    joint_names = ("shoulder", "elbow", "wrist")

    def reconstruct(
        self,
        pairs: list[SynchronizedFramePair] | tuple[SynchronizedFramePair, ...],
        sequence_id: str,
    ) -> ReconstructionSequence:
        timestamps = np.array(
            [pair.left.timestamp_ns for pair in pairs],
            dtype=np.int64,
        )
        frame_count = len(pairs)
        joints = np.zeros((frame_count, 3, 3), dtype=np.float64)
        phase = np.linspace(0.0, np.pi, frame_count) if frame_count else np.array([])
        joints[:, 0, :] = [0.0, 1.0, 0.0]
        joints[:, 1, :] = [0.0, 0.5, 0.0]
        joints[:, 2, 0] = np.cos(phase)
        joints[:, 2, 1] = 0.5 + np.sin(phase)
        validity = np.ones((frame_count, 3), dtype=bool)
        return ReconstructionSequence(
            sequence_id=sequence_id,
            timestamps_ns=timestamps,
            joints=joints,
            validity_mask=validity,
            joint_names=self.joint_names,
            coordinate_system="mock_world",
            units="metres",
            reconstruction_version="mock-reconstruction-v1",
            provenance=Provenance.MOCK_SYNTHETIC,
            source_pair_count=frame_count,
        )
