"""Train a record-disjoint CNN-LSTM with beat morphology and RR features."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score
from tensorflow.keras import callbacks, layers, models


DATASET = Path("data/processed_dataset.npz")
MODEL_PATH = Path("models/ecg_model.keras")
SPLIT_PATH = Path("data/dataset_split.npz")
SEED = 42
IMBALANCE_STRATEGY = "focal_only"


def focal_loss(alpha, gamma=2.0):
    alpha = tf.constant(alpha, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.one_hot(y_true, depth=len(alpha))
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0)
        p_t = tf.reduce_sum(one_hot * y_pred, axis=-1)
        alpha_t = tf.gather(alpha, y_true)
        return tf.reduce_mean(-alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t))

    return loss


def build_model(input_length, n_classes):
    beat = layers.Input((input_length, 1), name="beat")
    rr_features = layers.Input((3,), name="rr_features")
    x = layers.Conv1D(32, 7, padding="same", activation="relu")(beat)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(64)(x)
    x = layers.Concatenate()([x, layers.Dense(16, activation="relu")(rr_features)])
    output = layers.Dense(n_classes, activation="softmax")(layers.Dropout(0.4)(x))
    return models.Model({"beat": beat, "rr_features": rr_features}, output)


def grouped_splits(y, groups):
    if len(np.unique(groups)) < 3:
        raise ValueError("At least three distinct records are required for grouped splits.")
    train_val, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(y, y, groups)
    )
    train_rel, val_rel = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(
            train_val, y[train_val], groups[train_val]
        )
    )
    return train_val[train_rel], train_val[val_rel], test


def validate_splits(y, groups, splits):
    split_groups = [set(groups[index]) for index in splits]
    if any(split_groups[i] & split_groups[j] for i in range(3) for j in range(i)):
        raise AssertionError("Record leakage detected between dataset splits.")
    for name, index in zip(("train", "validation", "test"), splits):
        missing = sorted(set(range(int(y.max()) + 1)) - set(y[index]))
        if missing:
            print(f"Warning: {name} split has no samples for classes {missing}.")


class ValidationMacroF1(callbacks.Callback):
    def __init__(self, validation_inputs, validation_labels):
        super().__init__()
        self.validation_inputs = validation_inputs
        self.validation_labels = validation_labels

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        probabilities = self.model.predict(self.validation_inputs, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        score = f1_score(
            self.validation_labels,
            predictions,
            labels=np.arange(probabilities.shape[1]),
            average="macro",
            zero_division=0,
        )
        logs["val_macro_f1"] = float(score)
        print(f" - val_macro_f1: {score:.4f}")


def train():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    if not DATASET.exists():
        raise FileNotFoundError("Dataset not found. Run `python create_dataset.py` first.")
    data = np.load(DATASET)
    X = data["X"].astype("float32")
    rr = data["rr_features"].astype("float32")
    y = data["y"].astype("int64")
    groups = data["group_ids"].astype(str) if "group_ids" in data else data["record_ids"].astype(str)
    n_classes = len(data["class_names"])
    splits = grouped_splits(y, groups)
    validate_splits(y, groups, splits)
    np.savez_compressed(
        SPLIT_PATH,
        train=splits[0],
        validation=splits[1],
        test=splits[2],
        train_records=np.unique(data["record_ids"][splits[0]]),
        validation_records=np.unique(data["record_ids"][splits[1]]),
        test_records=np.unique(data["record_ids"][splits[2]]),
        train_groups=np.unique(groups[splits[0]]),
        validation_groups=np.unique(groups[splits[1]]),
        test_groups=np.unique(groups[splits[2]]),
    )
    mean = float(X[splits[0]].mean())
    std = float(X[splits[0]].std())
    if std <= 0:
        raise ValueError("Training beats have zero variance; cannot standardize the dataset.")
    X = ((X - mean) / std)[..., None]
    rr_mean = rr[splits[0]].mean(axis=0)
    rr_std = np.maximum(rr[splits[0]].std(axis=0), 1e-6)
    rr = (rr - rr_mean) / rr_std
    X_train, rr_train, y_train = X[splits[0]], rr[splits[0]], y[splits[0]]
    counts = np.maximum(np.bincount(y_train, minlength=n_classes).astype("float32"), 1)
    alpha = counts.sum() / (n_classes * counts)
    alpha = (alpha / alpha.mean()).tolist()
    model = build_model(X.shape[1], n_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=focal_loss(alpha),
        metrics=["accuracy"],
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        {"beat": X_train, "rr_features": rr_train},
        y_train,
        validation_data=(
            {"beat": X[splits[1]], "rr_features": rr[splits[1]]},
            y[splits[1]],
        ),
        epochs=100,
        batch_size=32,
        callbacks=[
            ValidationMacroF1(
                {"beat": X[splits[1]], "rr_features": rr[splits[1]]}, y[splits[1]]
            ),
            callbacks.EarlyStopping(
                monitor="val_macro_f1", mode="max", patience=10, restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                MODEL_PATH, monitor="val_macro_f1", mode="max", save_best_only=True
            ),
            callbacks.CSVLogger("training_history.csv"),
        ],
        verbose=1,
    )
    model.save(MODEL_PATH)
    MODEL_PATH.with_suffix(".json").write_text(
        json.dumps(
            {
                "class_names": data["class_names"].astype(str).tolist(),
                "n_classes": n_classes,
                "seed": SEED,
                "model_input_length": int(X.shape[1]),
                "inputs": {"beat": [int(X.shape[1]), 1], "rr_features": [3]},
                "filter": {"lowcut_hz": 0.5, "highcut_hz": 40.0, "order": 4},
                "excluded_records": ["102", "104", "107", "217"],
                "grouping": "patient_group; records 201 and 202 are kept together",
                "imbalance_strategy": IMBALANCE_STRATEGY,
                "rr_features": [
                    "pre_rr/local_median_rr",
                    "post_rr/local_median_rr",
                    "pre_rr/post_rr",
                ],
                "normalization": {"method": "training_global_zscore", "mean": mean, "std": std},
                "rr_normalization": {
                    "mean": rr_mean.tolist(),
                    "std": rr_std.tolist(),
                    "method": "training_only_zscore",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {MODEL_PATH}; test beats: {len(splits[2])}")


if __name__ == "__main__":
    train()
