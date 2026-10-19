"""SQLite metadata store for the local prototype."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from courtsense.domain.models import ConsentRecord, Session


class MetadataStore:
    """Owns metadata only; large arrays remain filesystem artifacts."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as db:
            db.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS participants (
                    participant_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'active'
                );
                CREATE TABLE IF NOT EXISTS consents (
                    consent_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    participant_id TEXT NOT NULL,
                    analysis_allowed INTEGER NOT NULL,
                    history_retention_allowed INTEGER NOT NULL,
                    training_retention_allowed INTEGER NOT NULL,
                    consent_version TEXT NOT NULL,
                    withdrawn_at TEXT,
                    FOREIGN KEY (participant_id) REFERENCES participants(participant_id)
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    participant_id TEXT,
                    pipeline_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    withdrawn_at TEXT,
                    FOREIGN KEY (participant_id) REFERENCES participants(participant_id)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    artifact_type TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL UNIQUE,
                    checksum TEXT NOT NULL,
                    processing_version TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active'
                );
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    dataset_version_id TEXT PRIMARY KEY,
                    manifest_artifact_id TEXT NOT NULL,
                    feature_artifact_id TEXT NOT NULL,
                    processing_version TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'finalized'
                );
                CREATE TABLE IF NOT EXISTS dataset_samples (
                    dataset_version_id TEXT NOT NULL,
                    shot_id TEXT NOT NULL,
                    participant_id TEXT NOT NULL,
                    split TEXT NOT NULL,
                    PRIMARY KEY (dataset_version_id, shot_id)
                );
                """
            )

    def register(self, session: Session, consent: ConsentRecord) -> None:
        if session.participant_id != consent.participant_id:
            raise ValueError("session and consent participant IDs must match")
        with self.connection() as db:
            db.execute(
                "INSERT OR IGNORE INTO participants(participant_id) VALUES (?)",
                (consent.participant_id,),
            )
            db.execute(
                """
                INSERT INTO consents(
                    participant_id, analysis_allowed, history_retention_allowed,
                    training_retention_allowed, consent_version
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    consent.participant_id,
                    int(consent.analysis_allowed),
                    int(consent.history_retention_allowed),
                    int(consent.training_retention_allowed),
                    consent.consent_version,
                ),
            )
            db.execute(
                """
                INSERT INTO sessions(session_id, participant_id, pipeline_version, status)
                VALUES (?, ?, ?, 'completed')
                ON CONFLICT(session_id) DO UPDATE SET
                    participant_id=excluded.participant_id,
                    pipeline_version=excluded.pipeline_version,
                    status='completed',
                    withdrawn_at=NULL
                """,
                (session.session_id, session.participant_id, session.pipeline_version),
            )

    def add_artifact(
        self, artifact_id: str, artifact_type: str, owner_id: str,
        relative_path: str, checksum: str, processing_version: str,
    ) -> None:
        with self.connection() as db:
            db.execute(
                """
                INSERT INTO artifacts(
                    artifact_id, artifact_type, owner_id, relative_path,
                    checksum, processing_version
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    checksum=excluded.checksum, status='active'
                """,
                (artifact_id, artifact_type, owner_id, relative_path, checksum, processing_version),
            )

    def active_training_sessions(self) -> list[sqlite3.Row]:
        with self.connection() as db:
            return db.execute(
                """
                SELECT DISTINCT s.session_id, s.participant_id
                FROM sessions s
                JOIN consents c ON c.participant_id = s.participant_id
                WHERE s.status = 'completed' AND s.withdrawn_at IS NULL
                  AND c.analysis_allowed = 1 AND c.training_retention_allowed = 1
                """
            ).fetchall()

    def withdraw_session(self, session_id: str) -> None:
        with self.connection() as db:
            db.execute(
                "UPDATE sessions SET status='withdrawn', withdrawn_at=datetime('now') "
                "WHERE session_id=?", (session_id,),
            )
            db.execute(
                "UPDATE artifacts SET status='withdrawn' WHERE owner_id=?", (session_id,),
            )

    def add_dataset_version(
        self, dataset_version_id: str, manifest_artifact_id: str,
        feature_artifact_id: str, processing_version: str,
        samples: list[tuple[str, str, str]],
    ) -> None:
        with self.connection() as db:
            db.execute(
                """
                INSERT INTO dataset_versions(
                    dataset_version_id, manifest_artifact_id, feature_artifact_id,
                    processing_version
                ) VALUES (?, ?, ?, ?)
                """,
                (dataset_version_id, manifest_artifact_id, feature_artifact_id, processing_version),
            )
            db.executemany(
                """
                INSERT INTO dataset_samples(dataset_version_id, shot_id, participant_id, split)
                VALUES (?, ?, ?, ?)
                """,
                [(dataset_version_id, shot, participant, split)
                 for shot, participant, split in samples],
            )
