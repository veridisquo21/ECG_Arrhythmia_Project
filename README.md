# ECG Arrhythmia Project

This repository contains an experimental MIT-BIH beat-classification pipeline. It applies a zero-phase band-pass filter, creates labeled beat segments, trains a CNN-LSTM model with morphology and RR-interval features, and evaluates it on record-disjoint data. The output is a **prediction**, not a medical diagnosis or clinical decision.

## Pipeline

```text
download_data.py -> create_dataset.py -> train_model.py -> evaluate_model.py
```

The dataset stores the originating MIT-BIH record, annotation symbol, sampling metadata, and RR features for every beat. Training creates separate train, validation, and test splits by record, so augmented training beats cannot leak into validation or test data. The test split is saved in `data/dataset_split.npz` and is the only split used by `evaluate_model.py`.

The default protocol uses the four AAMI classes N/S/V/F. Paced records `102`, `104`, `107`, and `217` are excluded because the Q/paced class is strongly confounded with patient identity. This is an experimental inter-patient baseline, not clinical validation.

## Installation

```bash
git clone https://github.com/veridisquo21/ECG_Arrhythmia_Project.git
cd ECG_Arrhythmia_Project
python -m pip install -r requirements.txt
```

## Usage

Download the complete MIT-BIH Arrhythmia Database, then build and train:

```bash
python download_data.py
python create_dataset.py
python train_model.py
python evaluate_model.py
python evaluate_baselines.py
```

The trained model is written to `models/ecg_model.keras` and its class metadata to `models/ecg_model.json`. The metadata records the filter, normalization, seed, inputs, and excluded records. To validate the simple peak detector against annotations:

```bash
python evaluate_group_kfold.py
python validate_detector.py
```

To run the unit tests:

```bash
pytest
```

The monitoring demo uses only the held-out test beats:

```bash
python advanced_monitor.py
```

The demo intentionally displays `BPM: N/A`: a single 300-sample beat does not contain enough consecutive R peaks for a valid heart-rate calculation. `main.py` calculates BPM only from a continuous signal with at least two detected peaks and rejects implausible RR intervals.

## Important limitations

- MIT-BIH record-disjoint evaluation is more realistic than beat-level random splitting, but it is not a clinical validation study.
- Signals are standardized using mean and standard deviation computed from the training split only; no test statistics are used.
- RR features are normalized ratios of the previous and following intervals to a wider local median RR context; their final z-score uses training-only statistics.
- The default peak detector is a transparent baseline. Run `validate_detector.py` before trusting it on a new record; `wfdb.processing.xqrs_detect` can be evaluated as a stronger alternative.
- The default training run uses class-weighted focal loss only. Oversampling and noise augmentation are not combined with focal weighting; these alternatives should be compared separately in an ablation study.
- Run `evaluate_baselines.py` to compare majority, morphology-only, and RR-only baselines on the same held-out split.
- Run `evaluate_group_kfold.py` to report patient-grouped five-fold baseline mean and standard deviation.
- Results can vary with the available patient groups and class distribution. Report macro-F1 and per-class recall, not accuracy alone.
- `training_history.csv` records loss, accuracy, and validation macro-F1 for plotting learning curves.

## Structure

```text
create_dataset.py       Build labeled beats, RR features, and preserve record IDs
train_model.py          Patient-group split, train-only standardization, training, model export
evaluate_model.py       Held-out test report and confusion matrix
evaluate_baselines.py   Majority, morphology-only, and RR-only baselines
validate_detector.py    Compare detected peaks with annotations
download_data.py        Download MIT-BIH records
advanced_monitor.py     Held-out beat visualization and prediction demo
simulation.py           Single-beat animation demo
src/data_loader.py      WFDB loading and lead selection
src/preprocess.py       Filtering, peak detection, BPM, segmentation, metrics
src/model_utils.py      Shared model loading and prediction preprocessing
```

MIT License. See the repository history for attribution and license details.
