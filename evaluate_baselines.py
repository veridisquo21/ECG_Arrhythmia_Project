"""Evaluate simple baselines on the same held-out record-disjoint test split."""

from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATASET = Path("data/processed_dataset.npz")
SPLIT_PATH = Path("data/dataset_split.npz")


def evaluate():
    if not DATASET.exists() or not SPLIT_PATH.exists():
        raise FileNotFoundError("Run create_dataset.py and train_model.py first.")
    data = np.load(DATASET)
    split = np.load(SPLIT_PATH)
    train, test = split["train"], split["test"]
    y_train, y_test = data["y"][train], data["y"][test]
    names = data["class_names"].astype(str).tolist()
    beats_train = data["X"][train]
    beats_test = data["X"][test]
    rr_train = data["rr_features"][train]
    rr_test = data["rr_features"][test]
    n_classes = len(names)

    baselines = {
        "majority": (
            DummyClassifier(strategy="most_frequent", random_state=42),
            beats_train.reshape(len(train), -1),
            beats_test.reshape(len(test), -1),
        ),
        "morphology_only": (
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=300, class_weight="balanced", random_state=42),
            ),
            beats_train.reshape(len(train), -1),
            beats_test.reshape(len(test), -1),
        ),
        "rr_only": (
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=300, class_weight="balanced", random_state=42),
            ),
            rr_train,
            rr_test,
        ),
    }
    for name, (estimator, train_features, test_features) in baselines.items():
        estimator.fit(train_features, y_train)
        predictions = estimator.predict(test_features)
        macro_f1 = f1_score(
            y_test,
            predictions,
            labels=np.arange(n_classes),
            average="macro",
            zero_division=0,
        )
        print(f"\n{name}: macro-F1={macro_f1:.4f}")
        print(
            classification_report(
                y_test, predictions, labels=np.arange(n_classes),
                target_names=names, zero_division=0,
            )
        )


if __name__ == "__main__":
    evaluate()
