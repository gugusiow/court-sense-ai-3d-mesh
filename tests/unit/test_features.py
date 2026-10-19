import numpy as np

from courtsense.domain.models import Provenance, ReconstructionSequence, Shot
from courtsense.features.extraction import extract_features


def test_zero_length_vector_is_invalid_not_an_exception():
    sequence = ReconstructionSequence(
        "sequence",
        np.array([1], dtype=np.int64),
        np.zeros((1, 3, 3)),
        np.ones((1, 3), dtype=bool),
        ("shoulder", "elbow", "wrist"),
        "mock",
        "metres",
        "test",
        Provenance.MOCK_SYNTHETIC,
        1,
    )
    features = extract_features(Shot("shot", "session", "sequence", 1, 1), sequence)
    angle = next(feature for feature in features if feature.feature_name == "mean_elbow_angle")
    assert angle.value is None
    assert not angle.valid
