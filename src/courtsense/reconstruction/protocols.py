"""Reconstruction integration boundary."""

from collections.abc import Sequence
from typing import Protocol

from courtsense.domain.models import (
    ReconstructionSequence,
    SynchronizedFramePair,
)


class ReconstructionAdapter(Protocol):
    def reconstruct(
        self,
        pairs: Sequence[SynchronizedFramePair],
        sequence_id: str,
    ) -> ReconstructionSequence:
        ...
