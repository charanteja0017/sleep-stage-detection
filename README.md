# Sleep Stage Detection from EEG

Five-class sleep staging (W / N1 / N2 / N3 / REM) from single-channel EEG,
using a compact 1D residual network in PyTorch with MNE-based preprocessing.

## Model

`SleepResNet1D` — **2,468,099 parameters** (verified: `python3 src/models/sleep_resnet.py`)

- Wide stem (kernel 49, stride 4) to cover low-frequency sleep rhythms, then max-pool
- Four residual stages, widths 34 → 272, stride-2 downsampling between stages
- Global average pooling → 5-way linear head
- Input: one 30 s epoch at 100 Hz = 3000 samples

## Status

| Component | State |
|---|---|
| Model, dataset, metrics, training loop | Working, run end-to-end |
| Pipeline validation on synthetic data | Passing, incl. shuffled-label negative control |
| Sleep-EDF preprocessing (`prepare_sleep_edf.py`) | Verified on generated EDF files (exact round-trip) |
| Full chain: `.edf` → preprocess → train → evaluate | Working |
| Results on **real PhysioNet** recordings | **Not yet produced** |

Preprocessing was checked by writing real EDF/EDF+ files in Sleep-EDF's format
and reading them back through the script: stage labels came back identical and
the EEG signal to within 0.003 µV (EDF's 16-bit quantisation floor). Subject
grouping was confirmed to pair both nights of each subject.

What that does *not* cover is the quirks of genuine PhysioNet files — channel
naming variations, irregular annotation spans, `Movement time` / `Sleep stage ?`
scores. Those need the real download.

Target for the real-data run: ~75.8% balanced accuracy, ~0.683 Cohen's kappa
across 153 recordings. Those are reference figures to reproduce, **not** numbers
this repository has yet measured.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Quick check (no data download)

Generates synthetic recordings with coarse per-stage spectral signatures and
trains on them. Useful for confirming the pipeline runs:

```bash
python3 src/utils/synthetic.py --out-dir data/processed --subjects 16
python3 src/train.py --epochs 4 --num-workers 0
```

The synthetic stages are deliberately easy and the model reaches ~1.00 on them;
this validates plumbing, not sleep-staging skill. The meaningful check is the
negative control — shuffle the labels and the model must collapse to chance:

```bash
python3 src/utils/synthetic.py --out-dir data/shuffled --subjects 16 --shuffle-labels
python3 src/train.py --data-dir data/shuffled --epochs 4 --num-workers 0
```

Measured: balanced accuracy 0.194, Cohen's kappa −0.009 — chance, as it should be.
If this run scores well, there is label leakage somewhere.

## Real data (Sleep-EDF Expanded)

Download the sleep-cassette set (153 PSG/Hypnogram pairs) from
PhysioNet into `data/raw/`, then:

```bash
python3 src/preprocessing/prepare_sleep_edf.py --raw-dir data/raw --out-dir data/processed
python3 src/train.py --epochs 40 --batch-size 128
```

Preprocessing picks the Fpz-Cz channel, resamples to 100 Hz, merges scoring
stages 3 and 4 into N3, and trims to ±30 min of wake around the sleep period —
without that trim the cassette files are overwhelmingly wake.

## Handling class imbalance

N1 is only ~10% of epochs while N2 is ~35%, so plain accuracy is misleading and
the reported metrics are balanced accuracy and Cohen's kappa. Two mechanisms,
both tunable from the CLI:

- `--sampler-scheme sqrt_inverse` — oversamples rare stages during training
- `--loss-weight-scheme effective` — effective-number-of-samples weighting
  (Cui et al. 2019), gentler than raw inverse frequency

## Splitting

`split_by_subject` partitions by **subject**, not by recording. Sleep-EDF gives
most subjects two nights; splitting per-recording would put the same sleeper in
both train and test and inflate results.

## Performance notes (measured on this machine, Apple M-series / MPS)

- **`torch.compile` is disabled off CUDA by default.** On MPS it ran ~50x slower
  per epoch (278 s vs 5.1 s) *and* the model never left chance accuracy.
  Override with `--force-compile`.
- **AMP is CUDA-only here.** bf16 autocast where supported (no GradScaler needed),
  fp16 + GradScaler otherwise. MPS autocast is skipped; CPU uses bf16.
- **`non_blocking=True` is gated to CUDA.** On MPS an async host→device copy could
  be read before it landed, producing garbage logits (absmax ~1e33) on a random
  subset of batches — silent numerical corruption, not a crash.

## Layout

```
src/
├── models/sleep_resnet.py          # SleepResNet1D
├── preprocessing/prepare_sleep_edf.py
├── utils/dataset.py                # subject-wise splits, sampling, weighting
├── utils/metrics.py                # balanced accuracy, kappa, per-class F1
├── utils/synthetic.py              # synthetic data for pipeline checks
└── train.py                        # training / eval entry point
```

## Metrics

`src/utils/metrics.py` implements balanced accuracy, Cohen's kappa, per-class F1
and the confusion matrix directly in NumPy. Checked against hand-computed values
and behavioural cases: perfect prediction → kappa 1.000; random → kappa ≈ 0;
constant-majority prediction → balanced accuracy exactly 0.200.
