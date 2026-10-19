"""Participant-disjoint deterministic dataset splits."""


def participant_splits(participant_ids: list[str]) -> dict[str, str]:
    unique = sorted(set(participant_ids))
    if len(unique) < 3:
        raise ValueError("at least three participants are required for disjoint splits")
    return {
        participant: ("train" if index == 0 else "validation" if index == 1 else "test")
        for index, participant in enumerate(unique)
    }
