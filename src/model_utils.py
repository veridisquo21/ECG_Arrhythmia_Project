"""Shared model metadata, prediction, and continuous-record utilities."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def load_model_bundle(model_path="models/ecg_model.keras"):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run train_model.py first.")
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}.")
    model = tf.keras.models.load_model(path, compile=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def predict_beats(model, beats, rr_features, metadata=None):
    beats = np.asarray(beats, dtype="float32")
    if metadata and metadata.get("normalization", {}).get("method") == "training_global_zscore":
        normalization = metadata["normalization"]
        beats = (beats - normalization["mean"]) / normalization["std"]
    if beats.ndim == 2:
        beats = beats[..., None]
    rr_features = np.asarray(rr_features, dtype="float32")
    if rr_features.ndim == 1:
        rr_features = rr_features.reshape(1, -1)
    if metadata and metadata.get("rr_normalization"):
        rr_norm = metadata["rr_normalization"]
        rr_features = (rr_features - np.asarray(rr_norm["mean"])) / np.asarray(
            rr_norm["std"]
        )
    return model.predict({"beat": beats, "rr_features": rr_features}, verbose=0)
