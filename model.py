"""
Traffic Light Recognition System using Deep Learning and IoT
Module: model.py
Description:
    CNN architecture definition, save/load helpers and single-image prediction.
    Architecture matches the trained file models/traffic_light_cnn.keras:
        Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Conv2D(64)
        -> GlobalAveragePooling2D -> Dense(64) -> Dropout(0.3) -> Dense(4, softmax)
"""

import os
from typing import Any, Dict, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # quieter TensorFlow logs
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU only (Render has no GPU)

import cv2
import numpy as np

CLASSES = ["RED", "YELLOW", "GREEN", "NONE"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = (64, 64)  # (width, height) used by cv2.resize

ACTION_ALERTS = {
    "RED": "STOP! Red signal active. Vehicle halt is mandatory.",
    "YELLOW": "CAUTION! Yellow signal active. Decelerate and prepare to stop.",
    "GREEN": "GO! Green signal active. Clear to proceed through intersection.",
    "NONE": "NO SIGNAL: No active traffic light identified.",
}


def build_traffic_light_cnn(input_shape=(64, 64, 3), num_classes: int = 4):
    """Builds and compiles the 3-block CNN with Global Average Pooling."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.30),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_traffic_model(model, path: str):
    """Saves the model in native Keras format."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    model.save(path)
    print(f"[MODEL] Saved model to: {path}")


class _LayerInfo:
    """Small description of one layer (used by the dashboard's architecture table)."""
    def __init__(self, name: str, kind: str, params: int):
        self.name = name
        self.kind = kind
        self._params = params

    def count_params(self) -> int:
        return self._params


class NumpyTrafficCNN:
    """
    Lightweight inference engine for the trained CNN, written in pure NumPy.

    It reads the weights straight from the .keras file, so TensorFlow is NOT
    needed to run the web app. This keeps RAM around ~200 MB, which fits on
    Render's free 512 MB instance (TensorFlow alone needs 800+ MB).
    Output is numerically identical to Keras (checked in test_numpy_engine.py).
    """

    def __init__(self, keras_path: str):
        import io
        import json
        import zipfile
        import h5py

        with zipfile.ZipFile(keras_path) as zf:
            config = json.loads(zf.read("config.json"))
            weights_bytes = zf.read("model.weights.h5")

        self.ops = []
        self.layers = []
        with h5py.File(io.BytesIO(weights_bytes), "r") as h5:
            for layer in config["config"]["layers"]:
                kind = layer["class_name"]
                cfg = layer["config"]
                name = cfg.get("name", kind)
                if kind == "InputLayer":
                    continue
                if kind in ("Conv2D", "Dense"):
                    w = np.asarray(h5[f"layers/{name}/vars/0"], dtype=np.float32)
                    b = np.asarray(h5[f"layers/{name}/vars/1"], dtype=np.float32)
                    self.ops.append((kind, w, b, cfg.get("activation", "linear")))
                    self.layers.append(_LayerInfo(name, kind, int(w.size + b.size)))
                elif kind in ("MaxPooling2D", "GlobalAveragePooling2D"):
                    self.ops.append((kind, None, None, None))
                    self.layers.append(_LayerInfo(name, kind, 0))
                elif kind == "Dropout":  # inactive at inference time
                    self.layers.append(_LayerInfo(name, kind, 0))
                else:
                    raise ValueError(f"Unsupported layer type in model: {kind}")

    def count_params(self) -> int:
        return sum(l.count_params() for l in self.layers)

    @staticmethod
    def _activate(x, act):
        if act == "relu":
            return np.maximum(x, 0.0)
        if act == "softmax":
            e = np.exp(x - x.max(axis=-1, keepdims=True))
            return e / e.sum(axis=-1, keepdims=True)
        return x

    @staticmethod
    def _conv_same(x, w, b):
        # x: (N, H, W, C), w: (kh, kw, C, F), stride 1, 'same' padding
        kh, kw = w.shape[:2]
        ph, pw = kh // 2, kw // 2
        xp = np.pad(x, ((0, 0), (ph, kh - 1 - ph), (pw, kw - 1 - pw), (0, 0)))
        win = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(1, 2))
        # win: (N, H, W, C, kh, kw)
        return np.einsum("nhwcij,ijcf->nhwf", win, w, optimize=True) + b

    @staticmethod
    def _maxpool2(x):
        n, h, w, c = x.shape
        x = x[:, : h // 2 * 2, : w // 2 * 2, :]
        return x.reshape(n, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))

    def __call__(self, x, training: bool = False):
        x = np.asarray(x, dtype=np.float32)
        for kind, w, b, act in self.ops:
            if kind == "Conv2D":
                x = self._activate(self._conv_same(x, w, b), act)
            elif kind == "MaxPooling2D":
                x = self._maxpool2(x)
            elif kind == "GlobalAveragePooling2D":
                x = x.mean(axis=(1, 2))
            elif kind == "Dense":
                x = self._activate(x @ w + b, act)
        return x

    def predict(self, x, verbose: int = 0):
        return self(x)


def load_traffic_model(path: str, use_tensorflow: Optional[bool] = None):
    """
    Loads the saved .keras model for inference.
    Default: lightweight NumPy engine (no TensorFlow needed).
    Set environment variable USE_TENSORFLOW=1 to load with Keras/TensorFlow instead.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    if use_tensorflow is None:
        use_tensorflow = os.environ.get("USE_TENSORFLOW", "0") == "1"
    if use_tensorflow:
        from tensorflow import keras
        return keras.models.load_model(path, compile=False)
    return NumpyTrafficCNN(path)


def preprocess_roi(roi_bgr: np.ndarray) -> np.ndarray:
    """BGR image -> (1, 64, 64, 3) float32 in [0, 1] (same as train.py)."""
    img = cv2.resize(roi_bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def predict_signal(model, roi_bgr: Optional[np.ndarray]) -> Dict[str, Any]:
    """
    Classifies one image region.
    Returns label, confidence (%), per-class probabilities (%) and an action alert.
    """
    if model is None or roi_bgr is None or roi_bgr.size == 0:
        probs = {c: 0.0 for c in CLASSES}
        probs["NONE"] = 100.0
        return {"label": "NONE", "confidence": 100.0 if model is not None else 0.0,
                "probabilities": probs, "action_alert": ACTION_ALERTS["NONE"]}

    if roi_bgr.ndim == 2:
        roi_bgr = cv2.cvtColor(roi_bgr, cv2.COLOR_GRAY2BGR)
    elif roi_bgr.shape[2] == 4:
        roi_bgr = cv2.cvtColor(roi_bgr, cv2.COLOR_BGRA2BGR)

    x = preprocess_roi(roi_bgr)
    # Direct call is faster and lighter than model.predict() for a single image
    out = model(x, training=False)
    p = np.asarray(out)[0]

    idx = int(np.argmax(p))
    label = CLASSES[idx]
    probabilities = {c: round(float(p[i]) * 100.0, 1) for i, c in enumerate(CLASSES)}
    return {
        "label": label,
        "confidence": round(float(p[idx]) * 100.0, 1),
        "probabilities": probabilities,
        "action_alert": ACTION_ALERTS[label],
    }
