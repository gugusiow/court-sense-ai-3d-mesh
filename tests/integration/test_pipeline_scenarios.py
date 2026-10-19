from courtsense.domain.models import ConsentRecord, QualityStatus, Session
from courtsense.pipeline.runner import run_synthetic_analysis


def test_no_retention_analysis_returns_summary_without_artifacts():
    result = run_synthetic_analysis(Session("s1", "p1"), ConsentRecord("p1"))
    assert result.quality.status is QualityStatus.PASS
    assert result.retained_artifact_ids == ()
    assert result.sequence.provenance.value == "mock_synthetic"


def test_history_retention_marks_derived_artifacts_but_not_training_consent():
    result = run_synthetic_analysis(
        Session("s2", "p2"),
        ConsentRecord("p2", history_retention_allowed=True),
    )
    assert len(result.retained_artifact_ids) == 2
    assert not ConsentRecord("p2", history_retention_allowed=True).training_retention_allowed


def test_training_retention_is_explicit():
    result = run_synthetic_analysis(
        Session("s3", "p3"),
        ConsentRecord("p3", training_retention_allowed=True),
    )
    assert result.retention_mode.value == "training"
    assert len(result.retained_artifact_ids) == 2
