"""Held-out beat prediction demo; BPM requires a continuous signal."""

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from src.model_utils import load_model_bundle, predict_beats


MODEL_PATH = Path("models/ecg_model.keras")
DATASET = Path("data/processed_dataset.npz")
SPLIT_PATH = Path("data/dataset_split.npz")
LOG_FILE = Path("Vaka_Kayitlari.csv")


def main():
    for path in (MODEL_PATH, DATASET, SPLIT_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run the data pipeline first.")
    model, metadata = load_model_bundle(MODEL_PATH)
    data = np.load(DATASET)
    test = np.load(SPLIT_PATH)["test"]
    X_test = data["X"][test].astype("float32")
    rr_test = data["rr_features"][test].astype("float32")
    record_ids = data["record_ids"][test].astype(str)
    rng = np.random.default_rng(42)
    selected = rng.choice(len(X_test), 4, replace=len(X_test) < 4)
    if not LOG_FILE.exists():
        with LOG_FILE.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                ["timestamp", "channel", "record_id", "bpm", "prediction", "confidence"]
            )

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=100)
    fig.suptitle("ECG prediction demo (not a diagnosis; one beat has no BPM)")
    lines, texts, bpm_texts = [], [], []
    for index, ax in enumerate(axes.flat):
        line, = ax.plot([], [], lw=2)
        ax.set_xlim(0, X_test.shape[1])
        ax.set_ylim(float(X_test.min()) - 0.05, float(X_test.max()) + 0.05)
        ax.set_title(f"Record: {record_ids[selected[index]]}")
        lines.append(line)
        texts.append(ax.text(5, float(X_test.max()), "Waiting for prediction"))
        bpm_texts.append(ax.text(190, float(X_test.max()), "BPM: N/A"))

    def update(frame):
        for channel, sample_index in enumerate(selected):
            lines[channel].set_data(np.arange(frame), X_test[sample_index][:frame])
        if frame == X_test.shape[1]:
            probabilities = predict_beats(model, X_test[selected], rr_test[selected], metadata)
            rows = []
            for channel, probability in enumerate(probabilities):
                class_index = int(np.argmax(probability))
                confidence = float(np.max(probability) * 100)
                texts[channel].set_text(
                    f"Prediction: {metadata['class_names'][class_index]} ({confidence:.1f}%)"
                )
                rows.append(
                    [
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        channel + 1,
                        record_ids[selected[channel]],
                        "N/A",
                        metadata["class_names"][class_index],
                        f"{confidence:.2f}%",
                    ]
                )
            with LOG_FILE.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)
        return lines + texts + bpm_texts

    FuncAnimation(fig, update, frames=X_test.shape[1] + 1, interval=10, blit=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
