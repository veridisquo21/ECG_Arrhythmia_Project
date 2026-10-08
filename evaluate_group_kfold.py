"""Estimate baseline variance with patient-grouped five-fold cross-validation."""

from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATASET = Path("data/processed_dataset.npz")


def evaluate():
    if not DATASET.exists():
        raise FileNotFoundError("Run create_dataset.py first.")
    data = np.load(DATASET)
    X = data["X"].astype("float32")
    rr = data["rr_features"].astype("float32")
    y = data["y"].astype("int64")
    groups = data["group_ids"].astype(str) if "group_ids" in data else data["record_ids"].astype(str)
    labels = np.arange(len(data["class_names"]))
    features = {
        "majority": (DummyClassifier(strategy="most_frequent", random_state=42), X.reshape(len(X), -1)),
        "morphology_only": (
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=300, class_weight="balanced", random_state=42),
            ),
            X.reshape(len(X), -1),
        ),
        "rr_only": (
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=300, class_weight="balanced", random_state=42),
            ),
            rr,
        ),
    }
    fold_count = min(5, len(np.unique(groups)))
    results = {name: [] for name in features}
    for train, test in GroupKFold(n_splits=fold_count).split(X, y, groups):
        for name, (estimator, values) in features.items():
            estimator.fit(values[train], y[train])
            prediction = estimator.predict(values[test])
            results[name].append(
                f1_score(
                    y[test], prediction, labels=labels, average="macro", zero_division=0
                )
            )
    for name, scores in results.items():
        print(f"{name}: " f"{np.mean(scores):.4f} +/- {np.std(scores):.4f} " f"{scores}")


if __name__ == "__main__":
    evaluate()
