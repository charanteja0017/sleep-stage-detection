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

        # Size everything up front and fill a preallocated array. Concatenating
        # a list of per-recording arrays instead would hold both copies at once,
        # which is ~3.7 GB at the full 153-recording scale.
        lengths, width = [], None
        for path in records:
            with np.load(path, allow_pickle=True) as d:
                lengths.append(int(d["y"].shape[0]))
                if width is None:
                    width = int(d["x"].shape[1])
        total = sum(lengths)

        self.x = np.empty((total, width), dtype=np.float32)
        self.y = np.empty(total, dtype=np.int64)
        self.subjects = np.empty(total, dtype=np.int64)

        at = 0
        for path, n in zip(records, lengths):
            with np.load(path, allow_pickle=True) as d:
                x = d["x"].astype(np.float32, copy=False)
                if normalize:
                    # per-recording robust scaling; sleep EEG amplitude varies a
                    # lot between subjects and electrode impedances.
                    med = np.float32(np.median(x))
                    iqr = np.float32(np.subtract(*np.percentile(x, [75, 25])))
                    x = (x - med) / (iqr + np.float32(1e-6))
                self.x[at:at + n] = x
                self.y[at:at + n] = d["y"]
                self.subjects[at:at + n] = int(d["subject"])
            at += n

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


class SleepSequenceDataset(Dataset):
    """Windows of L consecutive epochs, for sequence models.

    Windows never cross a recording boundary - a night's last epoch and the next
    night's first are not neighbours. Where a recording does not divide evenly by
    the stride the final window is padded and those positions are labelled -1, so
    the loss and the metrics can drop them rather than scoring invented epochs.
    """

    PAD = -1

    def __init__(self, records, seq_len=21, stride=None, normalize=True,
                 clip_uv=500.0, augment=False):
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.augment = augment

        base = SleepEpochDataset(records, normalize=normalize, clip_uv=clip_uv,
                                 augment=False)
        self.x, self.y, self.subjects = base.x, base.y, base.subjects

        bounds, at = [], 0
        for path in records:
            with np.load(path, allow_pickle=True) as d:
                n = int(d["y"].shape[0])
            bounds.append((at, at + n))
            at += n
        self.bounds = bounds

        self.windows = []
        for start, end in bounds:
            n = end - start
            if n == 0:
                continue
            pos = 0
            while pos < n:
                self.windows.append((start + pos, min(seq_len, n - pos)))
                pos += self.stride

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        start, valid = self.windows[i]
        L = self.seq_len
        xs = np.zeros((L, self.x.shape[1]), dtype=np.float32)
        ys = np.full(L, self.PAD, dtype=np.int64)
        xs[:valid] = self.x[start:start + valid]
        ys[:valid] = self.y[start:start + valid]

        if self.augment:
            if np.random.rand() < 0.5:
                xs[:valid] *= np.float32(np.random.uniform(0.8, 1.2))
            if np.random.rand() < 0.3:
                xs[:valid] = np.roll(xs[:valid], np.random.randint(-300, 300), axis=1)

        return torch.from_numpy(xs), torch.from_numpy(ys)

    def class_counts(self):
        return np.bincount(self.y, minlength=NUM_CLASSES)


def make_sequence_loaders(processed_dir, seq_len=21, batch_size=32, num_workers=0,
                          seed=42, train_stride=None, augment=True):
    """Same subject-wise split as make_loaders, windowed for a sequence model."""
    tr_f, va_f, te_f, subj_counts = split_by_subject(processed_dir, seed=seed)

    train = SleepSequenceDataset(tr_f, seq_len=seq_len, stride=train_stride,
                                 augment=augment)
    val = SleepSequenceDataset(va_f, seq_len=seq_len)
    test = SleepSequenceDataset(te_f, seq_len=seq_len)

    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                  persistent_workers=num_workers > 0)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              drop_last=True, **common)
    val_loader = DataLoader(val, batch_size=batch_size * 2, shuffle=False, **common)
    test_loader = DataLoader(test, batch_size=batch_size * 2, shuffle=False, **common)

    meta = {
        "subjects": dict(zip(["train", "val", "test"], subj_counts)),
        "recordings": {"train": len(tr_f), "val": len(va_f), "test": len(te_f)},
        "epochs": {"train": len(train.y), "val": len(val.y), "test": len(test.y)},
        "windows": {"train": len(train), "val": len(val), "test": len(test)},
        "train_class_counts": train.class_counts().tolist(),
        "seq_len": seq_len,
    }
    return train_loader, val_loader, test_loader, meta
