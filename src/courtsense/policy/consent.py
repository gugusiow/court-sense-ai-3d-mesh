"""Backend-enforced consent decisions."""

from courtsense.domain.models import ConsentRecord, RetentionMode


def retention_mode(consent: ConsentRecord) -> RetentionMode:
    if not consent.analysis_allowed:
        raise PermissionError("analysis is not permitted by the consent record")
    if consent.training_retention_allowed:
        return RetentionMode.TRAINING
    if consent.history_retention_allowed:
        return RetentionMode.HISTORY_ONLY
    return RetentionMode.NO_RETENTION


def can_retain_history(consent: ConsentRecord) -> bool:
    return consent.history_retention_allowed


def can_retain_training(consent: ConsentRecord) -> bool:
    return consent.training_retention_allowed
