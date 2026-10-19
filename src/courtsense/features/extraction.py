"""Small, explicitly defined features."""

import numpy as np

from courtsense.domain.models import FeatureRecord, ReconstructionSequence, Shot


def _joint_index(sequence: ReconstructionSequence, name: str) -> int:
    try:
        return sequence.joint_names.index(name)
    except ValueError as exc:
        raise ValueError(f"required joint is missing: {name}") from exc


def extract_features(shot: Shot, sequence: ReconstructionSequence) -> list[FeatureRecord]:
    shoulder = _joint_index(sequence, "shoulder")
    elbow = _joint_index(sequence, "elbow")
    wrist = _joint_index(sequence, "wrist")
    values: list[float] = []
    valid_angles = []
    for frame in range(len(sequence.timestamps_ns)):
        valid = sequence.validity_mask[frame, [shoulder, elbow, wrist]]
        upper = sequence.joints[frame, shoulder] - sequence.joints[frame, elbow]
        lower = sequence.joints[frame, wrist] - sequence.joints[frame, elbow]
        denominator = np.linalg.norm(upper) * np.linalg.norm(lower)
        if not valid.all() or denominator == 0:
            continue
        cosine = np.clip(np.dot(upper, lower) / denominator, -1.0, 1.0)
        valid_angles.append(float(np.degrees(np.arccos(cosine))))
    angle = float(np.mean(valid_angles)) if valid_angles else None
    values.append(angle if angle is not None else 0.0)
    wrist_valid = sequence.validity_mask[:, wrist]
    heights = sequence.joints[:, wrist, 1][wrist_valid]
    height_range = float(np.ptp(heights)) if len(heights) else None
    return [
        FeatureRecord(
            shot.shot_id,
            "mean_elbow_angle",
            angle,
            "degrees",
            angle is not None,
            1.0 - len(valid_angles) / max(len(sequence.timestamps_ns), 1),
            sequence.sequence_id,
        ),
        FeatureRecord(
            shot.shot_id,
            "wrist_height_range",
            height_range,
            "metres",
            height_range is not None,
            1.0 - len(heights) / max(len(sequence.timestamps_ns), 1),
            sequence.sequence_id,
        ),
    ]
