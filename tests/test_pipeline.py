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


def test_subject_ids_match_mne_manifest():
    """Subject id is parsed from the filename; if that parse is wrong the
    subject-wise split silently leaks a sleeper across train and test and every
    reported number is inflated. Checked against MNE's own manifest.

    Skipped when MNE is not installed.
    """
    import re

    try:
        import mne.datasets.sleep_physionet as sp
        import pandas as pd
    except ImportError:
        return

    csv = os.path.join(os.path.dirname(sp.__file__), "age_records.csv")
    if not os.path.exists(csv):
        return

    df = pd.read_csv(csv)
    psg = df[df["record type"] == "PSG"]
    assert len(psg) == 153

    for _, r in psg.iterrows():
        rec_id = r["fname"][:6]
        parsed = int(re.search(r"SC4(\d{2})", rec_id).group(1))
        assert parsed == r["subject"], (r["fname"], parsed, r["subject"])

    parsed_ids = {int(re.search(r"SC4(\d{2})", f[:6]).group(1)) for f in psg["fname"]}
    assert len(parsed_ids) == psg["subject"].nunique() == 78


def test_split_is_disjoint_by_subject():
    """No subject may appear in more than one split."""
    import tempfile

    from utils.dataset import split_by_subject

    with tempfile.TemporaryDirectory() as d:
        for subj in range(20):
            for night in (1, 2):
                np.savez(os.path.join(d, f"SC4{subj:02d}{night}.npz"),
                         x=np.zeros((4, 3000), np.float32), y=np.arange(4) % 5,
                         subject=subj, rec_id=f"SC4{subj:02d}{night}")
        tr, va, te, counts = split_by_subject(d, seed=0)

        subj_of = lambda fs: {int(np.load(f, allow_pickle=True)["subject"]) for f in fs}
        s_tr, s_va, s_te = subj_of(tr), subj_of(va), subj_of(te)
        assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)
        assert len(s_tr | s_va | s_te) == 20
        assert len(tr) + len(va) + len(te) == 40
        assert counts == (len(s_tr), len(s_va), len(s_te))


def test_sequence_windows_respect_recording_boundaries():
    """A window must never span two recordings - the last epoch of one night and
    the first of the next are not neighbours, and letting attention join them
    would invent temporal context that does not exist.
    """
    import tempfile

    from utils.dataset import SleepSequenceDataset

    L = 21
    with tempfile.TemporaryDirectory() as d:
        files, lengths = [], [50, 21, 13, 64]
        for i, n in enumerate(lengths):
            p_ = os.path.join(d, f"SC4{i:02d}1.npz")
            np.savez(p_, x=np.random.randn(n, 3000).astype(np.float32),
                     y=(np.arange(n) % 5), subject=i, rec_id=f"SC4{i:02d}1")
            files.append(p_)

        ds = SleepSequenceDataset(files, seq_len=L)

        for start, valid in ds.windows:
            rec = [b for b in ds.bounds if b[0] <= start < b[1]]
            assert len(rec) == 1
            assert start + valid <= rec[0][1], "window crossed a recording boundary"

        # every real epoch appears exactly once, in order
        seen = []
        for i in range(len(ds)):
            _, y = ds[i]
            y = y.numpy()
            seen.append(y[y != SleepSequenceDataset.PAD])
        assert np.array_equal(np.concatenate(seen), ds.y)


def test_sequence_padding_is_masked():
    """Short tails are padded; those positions must carry the ignore label so the
    loss and metrics never score an invented epoch."""
    import tempfile

    from utils.dataset import SleepSequenceDataset

    L = 21
    with tempfile.TemporaryDirectory() as d:
        n = 25                                    # one full window + a 4-epoch tail
        p_ = os.path.join(d, "SC4001.npz")
        np.savez(p_, x=np.random.randn(n, 3000).astype(np.float32),
                 y=(np.arange(n) % 5), subject=0, rec_id="SC4001")
        ds = SleepSequenceDataset([p_], seq_len=L)

        assert len(ds) == 2
        x, y = ds[1]
        assert int((y != SleepSequenceDataset.PAD).sum()) == 4
        assert int((y == SleepSequenceDataset.PAD).sum()) == L - 4
        assert bool((x[y == SleepSequenceDataset.PAD] == 0).all())


def test_flatten_sequence_drops_padding():
    """The training loop's masking must remove exactly the padded positions."""
    import torch

    from train import flatten_sequence

    logits = torch.randn(2, 5, 5)
    y = torch.tensor([[0, 1, 2, -1, -1], [3, 4, -1, -1, -1]])
    y_cpu = y.numpy().copy()
    fl, fy, fcpu = flatten_sequence(logits, y, y_cpu)
    assert fl.shape == (5, 5)
    assert fy.tolist() == [0, 1, 2, 3, 4]
    assert fcpu.tolist() == [0, 1, 2, 3, 4]

    # a non-sequence model's output passes through untouched
    flat = torch.randn(4, 5)
    ly = torch.tensor([0, 1, 2, 3])
    a, b, _ = flatten_sequence(flat, ly)
    assert a.shape == (4, 5) and b.tolist() == [0, 1, 2, 3]


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
