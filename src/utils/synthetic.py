"""Generate synthetic Sleep-EDF-shaped .npz files to smoke-test the pipeline.

Not a substitute for real data - stages carry only coarse spectral signatures
(delta in N3, spindles in N2, alpha in W) so a working model should score well
above chance and a broken one should not.
"""
import argparse
import os

import numpy as np

SFREQ = 100
EPOCH_LEN = 30 * SFREQ
CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]

# (band centre Hz, amplitude uV) per stage
STAGE_BANDS = {
    0: [(10.0, 22.0), (20.0, 10.0), (1.0, 5.0)],    # W: alpha + beta
    1: [(6.0, 18.0), (10.0, 8.0), (2.0, 8.0)],      # N1: theta, alpha dropout
    2: [(5.0, 20.0), (1.5, 16.0)],                  # N2: theta + some delta
    3: [(1.2, 62.0), (3.0, 14.0)],                  # N3: delta dominant
    4: [(6.5, 14.0), (14.0, 6.0), (1.0, 4.0)],      # REM: low-amplitude mixed
}
TRANSITIONS = np.array([
    [0.79, 0.12, 0.05, 0.00, 0.04],
    [0.22, 0.32, 0.38, 0.01, 0.07],
    [0.04, 0.07, 0.72, 0.11, 0.06],
    [0.02, 0.02, 0.20, 0.75, 0.01],
    [0.08, 0.08, 0.12, 0.00, 0.72],
])


def make_epoch(stage, rng, subject_gain):
    t = np.arange(EPOCH_LEN) / SFREQ
    x = rng.normal(0, 6.0, EPOCH_LEN)
    for freq, amp in STAGE_BANDS[stage]:
        f = freq * rng.uniform(0.85, 1.15)
        x += amp * rng.uniform(0.7, 1.3) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    if stage == 2 and rng.random() < 0.7:          # sleep spindle burst
        start = rng.integers(0, EPOCH_LEN - 150)
        win = np.hanning(150)
        x[start:start + 150] += 30 * win * np.sin(2 * np.pi * 13.5 * t[:150])
    if stage == 2 and rng.random() < 0.4:          # K-complex
        start = rng.integers(0, EPOCH_LEN - 100)
        x[start:start + 100] -= 55 * np.hanning(100)
    if stage == 0 and rng.random() < 0.3:          # movement artifact
        start = rng.integers(0, EPOCH_LEN - 200)
        x[start:start + 200] += rng.normal(0, 45, 200)
    return (x * subject_gain).astype(np.float32)


def make_recording(subject, rng, n_epochs):
    gain = rng.uniform(0.75, 1.35)                 # inter-subject amplitude spread
    stages, s = [], 0
    for _ in range(n_epochs):
        s = rng.choice(5, p=TRANSITIONS[s])
        stages.append(s)
    y = np.array(stages, dtype=np.int64)
    x = np.stack([make_epoch(int(s), rng, gain) for s in y])
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--subjects", type=int, default=20)
    ap.add_argument("--recordings-per-subject", type=int, default=2)
    ap.add_argument("--epochs-per-recording", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="destroy signal/label link - model should then score ~chance")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    total = np.zeros(5, dtype=np.int64)

    for subj in range(args.subjects):
        for r in range(args.recordings_per_subject):
            x, y = make_recording(subj, rng, args.epochs_per_recording)
            if args.shuffle_labels:
                y = rng.permutation(y)
            rec_id = f"SY{subj:03d}{r}"
            np.savez_compressed(os.path.join(args.out_dir, f"{rec_id}.npz"),
                                x=x, y=y, subject=subj, rec_id=rec_id)
            total += np.bincount(y, minlength=5)

    n = args.subjects * args.recordings_per_subject
    print(f"wrote {n} synthetic recordings ({total.sum():,} epochs) to {args.out_dir}")
    print("class totals:", dict(zip(CLASS_NAMES, total.tolist())))
    print("class balance:", {c: f"{v/total.sum():.1%}" for c, v in zip(CLASS_NAMES, total)})


if __name__ == "__main__":
    main()
