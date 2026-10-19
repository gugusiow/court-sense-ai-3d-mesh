"""Validation and quality reporting."""

import numpy as np

from courtsense.domain.models import (
    QualityCheck,
    QualityReport,
    QualityStatus,
    ReconstructionSequence,
    SynchronizationResult,
)


def validate_sequence(
    sequence: ReconstructionSequence,
    synchronization: SynchronizationResult | None = None,
) -> QualityReport:
    checks: list[QualityCheck] = []
    expected_shape = (len(sequence.timestamps_ns), len(sequence.joint_names), 3)
    checks.append(
        QualityCheck(
            "shape",
            sequence.joints.shape == expected_shape
            and sequence.validity_mask.shape == expected_shape[:2],
            "shape_invalid",
            "trajectory and validity-mask shapes are consistent",
        )
    )
    timestamps_valid = (
        sequence.timestamps_ns.ndim == 1
        and np.all(np.isfinite(sequence.timestamps_ns))
        and np.all(np.diff(sequence.timestamps_ns) > 0)
    )
    checks.append(
        QualityCheck(
            "timestamps",
            timestamps_valid,
            "timestamps_invalid",
            "timestamps are strictly increasing",
        )
    )
    checks.append(
        QualityCheck(
            "numeric_values",
            bool(np.all(np.isfinite(sequence.joints))),
            "values_invalid",
            "joint coordinates are finite",
        )
    )
    checks.append(
        QualityCheck(
            "metadata",
            bool(sequence.units and sequence.coordinate_system and sequence.joint_names),
            "metadata_missing",
            "units, coordinate system, and joint names are present",
        )
    )
    valid_ratio = float(sequence.validity_mask.mean()) if sequence.validity_mask.size else 0.0
    checks.append(
        QualityCheck(
            "validity_ratio",
            valid_ratio >= 0.8,
            "too_many_invalid_joints",
            f"valid joint observation ratio is {valid_ratio:.3f}",
        )
    )
    if synchronization is not None:
        checks.append(
            QualityCheck(
                "synchronization",
                bool(synchronization.pairs),
                "no_synchronized_pairs",
                f"{len(synchronization.pairs)} synchronized pairs available",
            )
        )
    status = (
        QualityStatus.PASS
        if all(check.passed for check in checks)
        else QualityStatus.REJECT
    )
    return QualityReport(sequence.sequence_id, status, tuple(checks))
