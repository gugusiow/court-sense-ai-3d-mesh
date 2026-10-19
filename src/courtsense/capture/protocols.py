"""Capture integration boundary."""

from collections.abc import Iterable
from typing import Protocol

from courtsense.domain.models import CameraConfiguration, FramePacket


class CaptureAdapter(Protocol):
    def capture(
        self,
        camera: CameraConfiguration,
        frame_count: int,
        start_timestamp_ns: int,
        frame_interval_ns: int,
    ) -> Iterable[FramePacket]:
        ...
