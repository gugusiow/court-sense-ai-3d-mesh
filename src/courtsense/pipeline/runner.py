"""M1 end-to-end in-memory pipeline."""

from dataclasses import dataclass

from courtsense.capture.synthetic import SyntheticCaptureAdapter
from courtsense.domain.models import (
    CameraConfiguration,
    ConsentRecord,
    FeatureRecord,
    QualityReport,
    ReconstructionSequence,
    RetentionMode,
    Session,
    Shot,
    SynchronizationResult,
)
from courtsense.features.extraction import extract_features
from courtsense.features.segmentation import segment_single_shot
from courtsense.policy.consent import retention_mode
from courtsense.quality.checks import validate_sequence
from courtsense.reconstruction.mock import MockReconstructionAdapter
from courtsense.synchronization.timestamp_pairing import pair_by_timestamp


@dataclass(frozen=True)
class AnalysisResult:
    session: Session
    retention_mode: RetentionMode
    synchronization: SynchronizationResult
    sequence: ReconstructionSequence
    shot: Shot
    quality: QualityReport
    features: tuple[FeatureRecord, ...]
    retained_artifact_ids: tuple[str, ...]


def run_synthetic_analysis(
    session: Session,
    consent: ConsentRecord,
    *,
    frame_count: int = 30,
    frame_interval_ns: int = 33_333_333,
    timestamp_tolerance_ns: int = 2_000_000,
) -> AnalysisResult:
    mode = retention_mode(consent)
    capture = SyntheticCaptureAdapter()
    left_camera = CameraConfiguration("camera-left", "left", 30.0)
    right_camera = CameraConfiguration("camera-right", "right", 30.0)
    left = capture.capture(left_camera, frame_count, 0, frame_interval_ns)
    right = capture.capture(right_camera, frame_count, 1_000_000, frame_interval_ns)
    synchronization = pair_by_timestamp(left, right, timestamp_tolerance_ns)
    sequence = MockReconstructionAdapter().reconstruct(
        synchronization.pairs,
        f"{session.session_id}:sequence-001",
    )
    quality = validate_sequence(sequence, synchronization)
    shot = segment_single_shot(sequence, session.session_id)
    features = tuple(extract_features(shot, sequence))
    retained = (
        (f"{session.session_id}:quality", f"{shot.shot_id}:features")
        if mode is not RetentionMode.NO_RETENTION
        else ()
    )
    return AnalysisResult(
        session,
        mode,
        synchronization,
        sequence,
        shot,
        quality,
        features,
        retained,
    )
