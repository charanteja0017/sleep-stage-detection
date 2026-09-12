"""Epoch dataset with subject-wise splitting and imbalance handling."""
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
NUM_CLASSES = 5


class SleepEpochDataset(Dataset):
    def __init__(self, records, normalize=True, clip_uv=500.0, augment=False):
        self.normalize = normalize
        self.clip_uv = clip_uv
        self.augment = augment

        xs, ys, subs = [], [], []
        for path in records:
            d = np.load(path, allow_pickle=True)
            x = d["x"].astype(np.float32)
            if normalize:
                # per-recording robust scaling; sleep EEG amplitude varies a lot
                # between subjects and electrode impedances.
                med = np.float32(np.median(x))
                iqr = np.float32(np.subtract(*np.percentile(x, [75, 25])))
                x = ((x - med) / (iqr + np.float32(1e-6))).astype(np.float32, copy=False)
            xs.append(x)
            ys.append(d["y"].astype(np.int64))
            subs.append(np.full(len(d["y"]), int(d["subject"]), dtype=np.int64))

        self.x = np.concatenate(xs)
        self.y = np.concatenate(ys)
        self.subjects = np.concatenate(subs)

        if clip_uv:
            np.clip(self.x, -clip_uv, clip_uv, out=self.x)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i]
        if self.augment:
            if np.random.rand() < 0.5:
                x = x * np.float32(np.random.uniform(0.8, 1.2))
            if np.random.rand() < 0.3:
                shift = np.random.randint(-300, 300)
                x = np.roll(x, shift)
        x = np.ascontiguousarray(x, dtype=np.float32)
        return torch.from_numpy(x).unsqueeze(0), int(self.y[i])

    def class_counts(self):
        return np.bincount(self.y, minlength=NUM_CLASSES)


def subject_of(path):
    d = np.load(path, allow_pickle=True)
    return int(d["subject"])


def split_by_subject(processed_dir, val_frac=0.15, test_frac=0.15, seed=42):
    """Split recordings so no subject appears in more than one split."""
    files = sorted(glob.glob(os.path.join(processed_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz files in {processed_dir} - run preprocessing first")

    by_subject = {}
    for f in files:
        by_subject.setdefault(subject_of(f), []).append(f)

    subjects = sorted(by_subject)
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))

    test_s = subjects[:n_test]
    val_s = subjects[n_test:n_test + n_val]
    train_s = subjects[n_test + n_val:]

    flat = lambda ss: [f for s in ss for f in by_subject[s]]
    return flat(train_s), flat(val_s), flat(test_s), (len(train_s), len(val_s), len(test_s))


def class_weights(counts, scheme="effective", beta=0.999):
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    if scheme == "effective":
        # Cui et al. 2019 - effective number of samples; less extreme than
        # raw inverse frequency, which over-weights N1 into instability.
        eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        w = 1.0 / eff
    elif scheme == "inverse":
        w = 1.0 / counts
    elif scheme == "sqrt_inverse":
        w = 1.0 / np.sqrt(counts)
    else:
        w = np.ones_like(counts)
    w = w / w.sum() * len(counts)
    return torch.tensor(w, dtype=torch.float32)


def balanced_sampler(labels, scheme="sqrt_inverse"):
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    per_class = 1.0 / np.sqrt(counts) if scheme == "sqrt_inverse" else 1.0 / counts
    weights = per_class[labels]
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(labels), replacement=True
    )


def make_loaders(processed_dir, batch_size=128, num_workers=4, seed=42,
                 sampler_scheme="sqrt_inverse", augment=True):
    tr_f, va_f, te_f, subj_counts = split_by_subject(processed_dir, seed=seed)

    train = SleepEpochDataset(tr_f, augment=augment)
    val = SleepEpochDataset(va_f)
    test = SleepEpochDataset(te_f)

    sampler = balanced_sampler(train.y, sampler_scheme) if sampler_scheme else None
    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                  persistent_workers=num_workers > 0)

    train_loader = DataLoader(train, batch_size=batch_size, sampler=sampler,
                              shuffle=sampler is None, drop_last=True, **common)
    val_loader = DataLoader(val, batch_size=batch_size * 2, shuffle=False, **common)
    test_loader = DataLoader(test, batch_size=batch_size * 2, shuffle=False, **common)

    meta = {
        "subjects": dict(zip(["train", "val", "test"], subj_counts)),
        "recordings": {"train": len(tr_f), "val": len(va_f), "test": len(te_f)},
        "epochs": {"train": len(train), "val": len(val), "test": len(test)},
        "train_class_counts": train.class_counts().tolist(),
    }
    return train_loader, val_loader, test_loader, meta
