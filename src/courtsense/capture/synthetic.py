"""Deterministic in-memory synthetic camera input."""

from courtsense.domain.models import CameraConfiguration, FramePacket


class SyntheticCaptureAdapter:
    """Generate metadata-only packets; no image payload is persisted."""

    def capture(
        self,
        camera: CameraConfiguration,
        frame_count: int,
        start_timestamp_ns: int = 0,
        frame_interval_ns: int = 33_333_333,
    ) -> list[FramePacket]:
        if frame_count < 0 or frame_interval_ns <= 0:
            raise ValueError("frame_count must be non-negative and interval must be positive")
        return [
            FramePacket(
                camera_id=camera.camera_id,
                timestamp_ns=start_timestamp_ns + index * frame_interval_ns,
                sequence_number=index,
                synthetic=True,
            )
            for index in range(frame_count)
        ]
