# Sleep Stage Detection EEG Classification

A deep learning system for automated sleep stage classification from EEG recordings using PyTorch.

## Project Overview

- **Architecture**: SleepResNet1D (2.5M parameters)
- **Performance**: 75.8% balanced accuracy, 0.683 Cohen's kappa
- **Dataset**: 153 EEG recordings across 5 sleep stages
- **Key Optimizations**: Mixed precision training, torch.compile graph compilation

## Sleep Stages
- W: Wake
- N1: Stage 1 (Light Sleep)
- N2: Stage 2 (Light Sleep)
- N3: Stage 3 (Deep Sleep)
- REM: REM Sleep

## Project Structure

```
Sleep stage detection/
├── data/
│   ├── raw/              # Original EEG recordings
│   └── processed/        # Preprocessed EEG data
├── src/
│   ├── models/           # Model architectures
│   ├── utils/            # Helper functions
│   └── preprocessing/    # Data preprocessing pipeline
├── notebooks/            # Jupyter notebooks for analysis
├── configs/              # Configuration files
├── results/              # Training results and logs
├── logs/                 # Training logs
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare data in `data/raw/` directory

## Usage

(Coming soon)

## References

- MNE-Python: https://mne.tools/
- PyTorch: https://pytorch.org/
