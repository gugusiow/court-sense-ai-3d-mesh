from courtsense.domain.models import ConsentRecord, RetentionMode
from courtsense.policy.consent import retention_mode


def test_history_and_training_consent_are_separate():
    assert retention_mode(ConsentRecord("p")) is RetentionMode.NO_RETENTION
    assert retention_mode(
        ConsentRecord("p", history_retention_allowed=True)
    ) is RetentionMode.HISTORY_ONLY
    assert retention_mode(
        ConsentRecord("p", training_retention_allowed=True)
    ) is RetentionMode.TRAINING
