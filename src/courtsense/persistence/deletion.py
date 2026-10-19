"""Withdrawal and deletion operations."""

from courtsense.persistence.artifacts import ArtifactStore
from courtsense.persistence.database import MetadataStore


def withdraw_session(metadata: MetadataStore, artifacts: ArtifactStore, session_id: str) -> None:
    metadata.withdraw_session(session_id)
    artifacts.delete_owner(session_id)
