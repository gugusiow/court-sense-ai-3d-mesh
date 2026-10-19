from courtsense.capture.synthetic import SyntheticCaptureAdapter
from courtsense.domain.models import CameraConfiguration
from courtsense.synchronization.timestamp_pairing import pair_by_timestamp


def test_pairs_by_timestamp_and_reports_unmatched_packets():
    capture = SyntheticCaptureAdapter()
    camera = CameraConfiguration("a", "a", 30.0)
    left = capture.capture(camera, 2, 0, 10)
    right = capture.capture(CameraConfiguration("b", "b", 30.0), 2, 3, 10)
    result = pair_by_timestamp(left, right, tolerance_ns=3)
    assert len(result.pairs) == 2
    assert not result.unmatched_left
    assert not result.unmatched_right


def test_frame_numbers_are_not_used_for_pairing():
    left = SyntheticCaptureAdapter().capture(
        CameraConfiguration("a", "a", 30.0), 1, 100, 10
    )
    right = SyntheticCaptureAdapter().capture(
        CameraConfiguration("b", "b", 30.0), 1, 101, 10
    )
    result = pair_by_timestamp(left, right, tolerance_ns=2)
    assert result.pairs[0].timestamp_delta_ns == 1
