"""Animate one held-out beat and show the model prediction."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from src.model_utils import load_model_bundle, predict_beats


def main():
    model_path = Path("models/ecg_model.keras")
    dataset_path = Path("data/processed_dataset.npz")
    split_path = Path("data/dataset_split.npz")
    if not all(path.exists() for path in (model_path, dataset_path, split_path)):
        raise FileNotFoundError("Run the data pipeline before starting the simulation.")
    model, metadata = load_model_bundle(model_path)
    data = np.load(dataset_path)
    test_index = np.load(split_path)["test"][0]
    sample = data["X"][test_index]
    rr = data["rr_features"][test_index]

    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot([], [], lw=2, color="#2ecc71")
    ax.set_xlim(0, len(sample))
    ax.set_ylim(float(sample.min()) - 0.05, float(sample.max()) + 0.05)
    ax.set_title("Held-out ECG beat — model prediction, not diagnosis")
    prediction = ax.text(10, float(sample.max()), "")

    def update(frame):
        line.set_data(np.arange(frame), sample[:frame])
        if frame == len(sample):
            probability = predict_beats(model, sample, rr, metadata)[0]
            index = int(np.argmax(probability))
            prediction.set_text(
                f"Prediction: {metadata['class_names'][index]} "
                f"({probability[index] * 100:.1f}%)"
            )
            line.set_color("#e74c3c" if index else "#2ecc71")
        return line, prediction

    FuncAnimation(fig, update, frames=len(sample) + 1, interval=10, blit=True, repeat=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
