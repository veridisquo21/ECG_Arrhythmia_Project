"""Create beat-level data while preserving the originating record for grouped splits."""

from pathlib import Path

import numpy as np

from src.data_loader import (
    AAMI_CLASSES,
    CLASS_NAMES,
    EXCLUDED_RECORDS,
    load_annotations,
    load_mit_bih_record,
    select_signal_channel,
)
from src.preprocess import apply_filters


DATA_DIR = Path("data/mitdb")
OUTPUT = Path("data/processed_dataset.npz")
WINDOW_SIZE = 150


def create_dataset(data_dir=DATA_DIR, output=OUTPUT):
    records = sorted(
        path.stem for path in Path(data_dir).glob("*.hea")
        if path.stem not in EXCLUDED_RECORDS
    )
    if not records:
        raise FileNotFoundError(
            f"No MIT-BIH records found in {data_dir}. Run download_data.py first."
        )

    beats, labels, record_ids, rr_features = [], [], [], []
    annotation_samples, annotation_symbols, lead_names, sampling_rates = [], [], [], []
    failures = []
    for record_id in records:
        path = Path(data_dir) / record_id
        try:
            signal, fields = load_mit_bih_record(str(path))
            samples, symbols = load_annotations(str(path))
            channel = select_signal_channel(signal, fields["sig_name"])
            lead_name = next(
                (name for name in ("MLII", "II", "V5") if name in fields["sig_name"]),
                fields["sig_name"][0],
            )
            filtered = apply_filters(channel, fields["fs"])
            for index, (sample, symbol) in enumerate(zip(samples, symbols)):
                label = AAMI_CLASSES.get(symbol)
                if label is None:
                    continue
                start, end = sample - WINDOW_SIZE, sample + WINDOW_SIZE
                if start >= 0 and end <= len(filtered):
                    beats.append(filtered[start:end])
                    labels.append(label)
                    record_ids.append(record_id)
                    annotation_samples.append(int(sample))
                    annotation_symbols.append(symbol)
                    lead_names.append(lead_name)
                    sampling_rates.append(float(fields["fs"]))
                    previous = (sample - samples[index - 1]) / fields["fs"] if index else 0.0
                    following = (
                        (samples[index + 1] - sample) / fields["fs"]
                        if index + 1 < len(samples)
                        else previous
                    )
                    local = np.median(
                        np.diff(samples[max(0, index - 2): min(len(samples), index + 3)])
                    ) / fields["fs"] if index > 0 else following
                    rr_features.append([previous, following, local])
        except (OSError, ValueError, RuntimeError) as error:
            failures.append(f"{record_id}: {error}")

    if not beats:
        raise RuntimeError("No labeled beats were created from the available records.")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X=np.asarray(beats, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int64),
        record_ids=np.asarray(record_ids),
        annotation_samples=np.asarray(annotation_samples, dtype=np.int64),
        annotation_symbols=np.asarray(annotation_symbols),
        lead_names=np.asarray(lead_names),
        sampling_rates=np.asarray(sampling_rates, dtype=np.float32),
        rr_features=np.asarray(rr_features, dtype=np.float32),
        class_names=np.asarray(CLASS_NAMES),
        dataset_version="1.0",
        filter_lowcut=0.5,
        filter_highcut=40.0,
        sampling_note="Beat-level RR features are seconds; excluded paced records are documented in src.data_loader.",
    )
    print(f"Created {len(beats)} beats from {len(records)} records.")
    print("Class counts:", dict(zip(*np.unique(labels, return_counts=True))))
    if failures:
        print("Failed records:")
        print("\n".join(failures))


if __name__ == "__main__":
    create_dataset()
