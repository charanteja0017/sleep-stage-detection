"""Regression tests. Run: python3 -m pytest tests/ -q  (or: python3 tests/test_pipeline.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from preprocessing.prepare_sleep_edf import STAGE_MAP
from utils.metrics import balanced_accuracy, cohen_kappa, confusion_matrix, summarize


def test_stage_event_codes_are_unique():
    """Scoring stages 3 and 4 both mean N3, but MNE needs distinct event codes.

    Mapping description -> class directly produced duplicate values and made
    events_from_annotations reject every real recording.
    """
    desc_to_code = {d: i + 1 for i, d in enumerate(STAGE_MAP)}
    assert len(set(desc_to_code.values())) == len(STAGE_MAP)

    code_to_class = {i + 1: STAGE_MAP[d] for i, d in enumerate(STAGE_MAP)}
    assert code_to_class[desc_to_code["Sleep stage 3"]] == 3
    assert code_to_class[desc_to_code["Sleep stage 4"]] == 3
    assert desc_to_code["Sleep stage 3"] != desc_to_code["Sleep stage 4"]
    assert set(code_to_class.values()) == {0, 1, 2, 3, 4}


def test_unscored_stages_are_excluded():
    """Real hypnograms contain 'Sleep stage ?' and 'Movement time'."""
    assert "Sleep stage ?" not in STAGE_MAP
    assert "Movement time" not in STAGE_MAP


def test_metrics_detect_failure():
    """A metric that cannot fail is worthless - pin the degenerate cases."""
    rng = np.random.default_rng(0)
    y = rng.choice(5, size=3000, p=[.25, .07, .42, .15, .11])

    assert summarize(y, y)["cohen_kappa"] == 1.0
    assert abs(summarize(y, rng.choice(5, size=3000))["cohen_kappa"]) < 0.05

    majority = np.full_like(y, 2)
    m = summarize(y, majority)
    assert abs(m["balanced_accuracy"] - 0.2) < 1e-9
    assert abs(m["cohen_kappa"]) < 1e-9
    assert m["accuracy"] > 0.35  # plain accuracy is fooled; that is the point


def test_kappa_matches_hand_computation():
    cm = np.array([[20, 5], [10, 15]])
    po, pe = 35 / 50, ((30 * 25) + (20 * 25)) / 2500
    assert abs(cohen_kappa(cm) - (po - pe) / (1 - pe)) < 1e-12


def test_balanced_accuracy_ignores_absent_classes():
    cm = confusion_matrix([0, 0, 1, 1], [0, 0, 1, 1], num_classes=5)
    assert abs(balanced_accuracy(cm) - 1.0) < 1e-12


def test_dataset_stays_float32():
    """np.median/np.percentile return float64 and silently promote the epochs,
    which MPS refuses outright."""
    import tempfile

    from utils.dataset import SleepEpochDataset

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "SY0000.npz")
        np.savez(p, x=(np.random.randn(8, 3000) * 40).astype(np.float32),
                 y=np.arange(8) % 5, subject=0, rec_id="SY0000")
        for augment in (False, True):
            x, _ = SleepEpochDataset([p], augment=augment)[0]
            assert x.dtype == __import__("torch").float32, augment
            assert x.shape == (1, 3000)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)} passed")
