"""Evaluate only the held-out record-disjoint test split."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.model_utils import load_model_bundle, predict_beats


DATASET = Path("data/processed_dataset.npz")
SPLIT_PATH = Path("data/dataset_split.npz")


def evaluate(show_plot=True):
    if not DATASET.exists() or not SPLIT_PATH.exists():
        raise FileNotFoundError("Run create_dataset.py and train_model.py first.")
    model, metadata = load_model_bundle()
    data = np.load(DATASET)
    test = np.load(SPLIT_PATH)["test"]
    y = data["y"][test].astype("int64")
    probabilities = predict_beats(
        model, data["X"][test], data["rr_features"][test], metadata
    )
    predicted = np.argmax(probabilities, axis=1)
    names = metadata["class_names"]
    labels = np.arange(len(names))
    report = classification_report(
        y, predicted, labels=labels, target_names=names, zero_division=0
    )
    print(report)
    print(f"Macro-F1: {f1_score(y, predicted, labels=labels, average='macro', zero_division=0):.4f}")
    if show_plot:
        sns.heatmap(
            confusion_matrix(y, predicted, labels=labels),
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=names,
            yticklabels=names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()
    return report


if __name__ == "__main__":
    evaluate()
