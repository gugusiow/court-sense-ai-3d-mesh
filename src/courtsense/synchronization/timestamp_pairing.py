"""Simple nearest-neighbour timestamp pairing."""

from courtsense.domain.models import (
    FramePacket,
    SynchronizedFramePair,
    SynchronizationResult,
)


def pair_by_timestamp(
    left: list[FramePacket],
    right: list[FramePacket],
    tolerance_ns: int,
) -> SynchronizationResult:
    if tolerance_ns < 0:
        raise ValueError("tolerance_ns must be non-negative")
    remaining_right = list(right)
    pairs: list[SynchronizedFramePair] = []
    unmatched_left: list[FramePacket] = []
    for left_packet in left:
        if not remaining_right:
            unmatched_left.append(left_packet)
            continue
        candidate = min(
            remaining_right,
            key=lambda packet: abs(packet.timestamp_ns - left_packet.timestamp_ns),
        )
        delta = abs(candidate.timestamp_ns - left_packet.timestamp_ns)
        if delta <= tolerance_ns:
            pairs.append(SynchronizedFramePair(left_packet, candidate, delta))
            remaining_right.remove(candidate)
        else:
            unmatched_left.append(left_packet)
    return SynchronizationResult(
        pairs=tuple(pairs),
        unmatched_left=tuple(unmatched_left),
        unmatched_right=tuple(remaining_right),
        tolerance_ns=tolerance_ns,
    )
