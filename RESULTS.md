# Results

Full Sleep-EDF Expanded sleep-cassette set: **153 recordings, 78 subjects, 195,469 thirty-second epochs**. Split by subject (54/12/12), so no sleeper appears on more than one side.

## Headline

| metric | measured | reference target |
|---|---|---|
| Cohen's kappa | **0.6837** | 0.683 |
| balanced accuracy | **0.7137** | 0.758 |
| accuracy | 0.7639 | — |
| macro F1 | 0.7015 | — |

**Kappa reproduces the target**: 0.6837 against 0.683.
**Balanced accuracy falls 0.044 short** of 0.758, and that gap is real, not noise —
validation kappa held between 0.716 and 0.730 over the last 8 epochs and balanced
accuracy between 0.686 and 0.696, so the run is stable. Most of the shortfall is N1
(F1 0.373); balanced accuracy is mean per-class recall, so the rarest and most
ambiguous stage drags it directly. Plausible causes not yet tested: longer training (this
stopped at epoch 28, best at 20), or temporal context across neighbouring epochs,
which this single-epoch model has none of.

## Per stage (test)

| stage | F1 | support |
|---|---|---|
| W | 0.921 | 8,309 |
| N1 | 0.373 | 3,062 |
| N2 | 0.776 | 9,969 |
| N3 | 0.716 | 2,748 |
| REM | 0.721 | 3,776 |

N1 at 0.373 is the weak point, as in all published single-channel work — it is a transition
stage human scorers themselves disagree on. The confusion structure below is physiologically
coherent: N1 scatters into W, N2 and REM; REM is taken for N1 and N2; N3 is confused almost
only with N2, never with wake.

### Confusion matrix (rows = true, cols = predicted)

| | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| **W** | 7,682 | 383 | 92 | 9 | 143 |
| **N1** | 474 | 1,073 | 975 | 113 | 427 |
| **N2** | 93 | 895 | 7,586 | 971 | 424 |
| **N3** | 11 | 15 | 466 | 2,255 | 1 |
| **REM** | 106 | 325 | 452 | 205 | 2,688 |

## Imbalance handling

Measured over 30 subjects, 15 epochs each. A balanced sampler and a class-weighted loss
correct the same skew, so running both boosts rare stages twice.

| sampler | loss weighting | balanced acc | kappa |
|---|---|---|---|
| `sqrt_inverse` | `effective` | 0.7254 | 0.7103 |
| `none` | `effective` | **0.7215** | **0.7404** |
| `sqrt_inverse` | `none` | 0.7210 | 0.7052 |

Weighted loss alone wins — same balanced accuracy, **+0.030 kappa**. `--sampler-scheme`
therefore defaults to `none`, and the full run above uses weighted loss only.

## Negative control

Identical pipeline, labels shuffled. If this scored above chance there would be leakage:

- balanced accuracy **0.194** (chance 0.200)
- Cohen's kappa **-0.009** (chance 0.000)

## Setup

- 2,468,099 parameters, single Fpz-Cz channel, no temporal context between epochs
- device `mps`, batch 128, lr 0.003, AdamW, cosine schedule with warmup
- loss weighting `effective`, sampler `none`, label smoothing 0.05
- trained 28 epochs (early stop, patience 8), best at 20 on validation balanced accuracy
- 133,319 train / 34,286 val / 27,864 test epochs

Raw report: [`results/full153_report.json`](results/full153_report.json)
