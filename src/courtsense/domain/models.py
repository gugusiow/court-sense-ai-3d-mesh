"""Small, explicit domain models for the M1 pipeline."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class Provenance(str, Enum):
    MOCK_SYNTHETIC = "mock_synthetic"
    REAL_RECONSTRUCTION = "real_reconstruction"


class RetentionMode(str, Enum):
    NO_RETENTION = "no_retention"
    HISTORY_ONLY = "history_only"
    TRAINING = "training"


class QualityStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    REJECT = "reject"


@dataclass(frozen=True)
class ConsentRecord:
    participant_id: str
    analysis_allowed: bool = True
    history_retention_allowed: bool = False
    training_retention_allowed: bool = False
    consent_version: str = "consent-v1"


@dataclass(frozen=True)
class Session:
    session_id: str
    participant_id: str | None
    pipeline_version: str = "pipeline-v1"


@dataclass(frozen=True)
class CameraConfiguration:
    camera_id: str
    view_name: str
    nominal_rate_hz: float
    configuration_version: str = "camera-v1"


@dataclass(frozen=True)
class FramePacket:
    camera_id: str
    timestamp_ns: int
    sequence_number: int
    synthetic: bool = True
    payload: Any = None


@dataclass(frozen=True)
class SynchronizedFramePair:
    left: FramePacket
    right: FramePacket
    timestamp_delta_ns: int


@dataclass(frozen=True)
class SynchronizationResult:
    pairs: tuple[SynchronizedFramePair, ...]
    unmatched_left: tuple[FramePacket, ...]
    unmatched_right: tuple[FramePacket, ...]
    tolerance_ns: int


@dataclass
class ReconstructionSequence:
    sequence_id: str
    timestamps_ns: np.ndarray
    joints: np.ndarray
    validity_mask: np.ndarray
    joint_names: tuple[str, ...]
    coordinate_system: str
    units: str
    reconstruction_version: str
    provenance: Provenance
    source_pair_count: int


@dataclass(frozen=True)
class QualityCheck:
    name: str
    passed: bool
    reason_code: str
    message: str


@dataclass(frozen=True)
class QualityReport:
    sequence_id: str
    status: QualityStatus
    checks: tuple[QualityCheck, ...]
    quality_version: str = "quality-v1"

    @property
    def eligible_for_training(self) -> bool:
        return self.status is QualityStatus.PASS


@dataclass(frozen=True)
class Shot:
    shot_id: str
    session_id: str
    sequence_id: str
    start_timestamp_ns: int
    end_timestamp_ns: int
    segment_version: str = "segment-v1"


@dataclass(frozen=True)
class FeatureRecord:
    shot_id: str
    feature_name: str
    value: float | None
    units: str
    valid: bool
    missing_data_fraction: float
    source_sequence_id: str
    feature_version: str = "features-v1"
