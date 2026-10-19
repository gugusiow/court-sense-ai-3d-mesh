"""Deterministic segmentation used only by the synthetic prototype."""

from courtsense.domain.models import ReconstructionSequence, Shot


def segment_single_shot(sequence: ReconstructionSequence, session_id: str) -> Shot:
    if not len(sequence.timestamps_ns):
        raise ValueError("cannot segment an empty sequence")
    return Shot(
        shot_id=f"{session_id}:shot-001",
        session_id=session_id,
        sequence_id=sequence.sequence_id,
        start_timestamp_ns=int(sequence.timestamps_ns[0]),
        end_timestamp_ns=int(sequence.timestamps_ns[-1]),
    )
