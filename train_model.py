"""Train a record-disjoint CNN-LSTM with beat morphology and RR features."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras import callbacks, layers, models


DATASET = Path("data/processed_dataset.npz")
MODEL_PATH = Path("models/ecg_model.keras")
SPLIT_PATH = Path("data/dataset_split.npz")
SEED = 42


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


def augment_training_data(X, rr, y):
    rng = np.random.default_rng(SEED)
    counts = np.bincount(y, minlength=int(y.max()) + 1)
    target = max(counts.max(), 1)
    extra_X, extra_rr, extra_y = [X], [rr], [y]
    for cls, count in enumerate(counts):
        if count == 0:
            continue
        amount = max(0, int(target - count))
        selected = rng.choice(np.flatnonzero(y == cls), amount, replace=True)
        extra_X.append(X[selected] + rng.normal(0, 0.01, X[selected].shape))
        extra_rr.append(rr[selected])
        extra_y.append(np.full(amount, cls, dtype=y.dtype))
    return np.concatenate(extra_X), np.concatenate(extra_rr), np.concatenate(extra_y)


def train():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    if not DATASET.exists():
        raise FileNotFoundError("Dataset not found. Run `python create_dataset.py` first.")
    data = np.load(DATASET)
    X = data["X"].astype("float32")
    rr = data["rr_features"].astype("float32")
    y = data["y"].astype("int64")
    groups = data["record_ids"].astype(str)
    n_classes = len(data["class_names"])
    splits = grouped_splits(y, groups)
    validate_splits(y, groups, splits)
    np.savez_compressed(SPLIT_PATH, train=splits[0], validation=splits[1], test=splits[2])
    mean = float(X[splits[0]].mean())
    std = float(X[splits[0]].std())
    if std <= 0:
        raise ValueError("Training beats have zero variance; cannot standardize the dataset.")
    X = ((X - mean) / std)[..., None]
    X_train, rr_train, y_train = augment_training_data(X[splits[0]], rr[splits[0]], y[splits[0]])
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
            callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
            callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
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
                "normalization": {"method": "training_global_zscore", "mean": mean, "std": std},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {MODEL_PATH}; test beats: {len(splits[2])}")


if __name__ == "__main__":
    train()
