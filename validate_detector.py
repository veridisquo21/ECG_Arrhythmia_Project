"""Validate the R-peak detector against MIT-BIH annotations."""

from pathlib import Path

import numpy as np

from src.data_loader import (
    AAMI_CLASSES,
    load_annotations,
    load_mit_bih_record,
    select_signal_channel,
)
from src.preprocess import apply_filters, detect_r_peaks, peak_detection_metrics


def validate(data_dir="data/mitdb"):
    reports = []
    for header in sorted(Path(data_dir).glob("*.hea")):
        record = header.stem
        signal, fields = load_mit_bih_record(str(Path(data_dir) / record))
        samples, symbols = load_annotations(str(Path(data_dir) / record))
        samples = np.asarray(
            [sample for sample, symbol in zip(samples, symbols) if symbol in AAMI_CLASSES]
        )
        filtered = apply_filters(
            select_signal_channel(signal, fields["sig_name"]), fields["fs"]
        )
        detected = detect_r_peaks(filtered, fields["fs"])
        report = peak_detection_metrics(detected, samples, fields["fs"])
        reports.append(report)
        print(
            f"{record}: sensitivity={report['sensitivity']:.3f}, "
            f"ppv={report['ppv']:.3f}, detected={len(detected)}"
        )
    if not reports:
        raise FileNotFoundError(f"No records found in {data_dir}.")
    print(
        f"Mean sensitivity: {np.mean([r['sensitivity'] for r in reports]):.3f}; "
        f"mean PPV: {np.mean([r['ppv'] for r in reports]):.3f}"
    )


if __name__ == "__main__":
    validate()
