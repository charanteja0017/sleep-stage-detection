# Results

## Status: preliminary — not the full dataset

The full 153-recording run has **not** been completed. What follows is a
30-recording subset (16 subjects), which is too small to quote as a result. Read the caveat below before using any number here.

## The headline numbers, and why they are not trustworthy yet

| metric | test | validation (range over training) |
|---|---|---|
| balanced accuracy | **0.818** | 0.486 – 0.679 |
| Cohen's kappa | **0.814** | 0.339 – 0.552 |
| accuracy | 0.863 | — |
| macro F1 | 0.807 | — |

The split put **2 subjects in validation and 2 in test**. Validation kappa sat around 0.50 while test came out at 0.81. A gap that wide between two tiny evaluation sets means the test subjects happened to be easy sleepers — it is not evidence the model is strong. The true figure lies somewhere between them, and pinning it down needs the full 78-subject set.

For reference, the target being reproduced is ~0.758 balanced accuracy and ~0.683 kappa across all 153 recordings.

## Per-stage breakdown (test)

| stage | F1 | support |
|---|---|---|
| W | 0.927 | 507 |
| N1 | 0.459 | 187 |
| N2 | 0.906 | 1,178 |
| N3 | 0.930 | 442 |
| REM | 0.813 | 497 |

N1 is the hard class here, as it is in every published sleep-staging result — it is a
transition stage that human scorers themselves disagree on. Single-channel models
typically land in the 0.40–0.50 F1 range for it. The rest of the confusion structure is
physiologically sensible: N1 scatters into W, N2 and REM; REM is mistaken for N1;
N3 separates almost cleanly because slow waves are distinctive.

### Confusion matrix (rows = true, cols = predicted)

| | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| **W** | 472 | 34 | 1 | 0 | 0 |
| **N1** | 30 | 103 | 29 | 0 | 25 |
| **N2** | 4 | 52 | 1,053 | 48 | 21 |
| **N3** | 2 | 0 | 14 | 426 | 0 |
| **REM** | 3 | 73 | 49 | 0 | 372 |

## Negative control

Same pipeline, labels shuffled. If this scored well, there would be leakage somewhere:

- balanced accuracy **0.194** (chance = 0.200)
- Cohen's kappa **-0.009** (chance = 0.000)

It collapses to chance, as it must.

## Setup

- 2,468,099 parameters
- device `mps`, batch 128, lr 0.003
- sampler `sqrt_inverse`, loss weighting `effective`
- best epoch 8, selected on validation balanced accuracy
- splits: 12/2/2 subjects, 24,193/3,926/2,811 epochs

Raw reports: [`results/real_subset_report.json`](results/real_subset_report.json), [`results/shuffled_report.json`](results/shuffled_report.json)
