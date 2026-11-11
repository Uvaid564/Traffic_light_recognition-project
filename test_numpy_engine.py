"""
Checks that the lightweight NumPy inference engine (used on Render) gives the
same output as the original Keras/TensorFlow model.
Requires TensorFlow (local only):  pip install -r requirements-train.txt
"""
import os
import sys

import cv2
import numpy as np

from model import load_traffic_model, preprocess_roi

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE, "models", "traffic_light_cnn.keras")

np_model = load_traffic_model(MODEL, use_tensorflow=False)
tf_model = load_traffic_model(MODEL, use_tensorflow=True)

batch = []
for folder in ("sample_test_images", "dataset/red", "dataset/yellow", "dataset/green", "dataset/none"):
    d = os.path.join(BASE, folder)
    if os.path.isdir(d):
        for f in sorted(os.listdir(d))[:25]:
            img = cv2.imread(os.path.join(d, f))
            if img is not None:
                batch.append(preprocess_roi(img)[0])
x = np.stack(batch)

p_np = np_model(x)
p_tf = np.asarray(tf_model(x, training=False))
max_diff = float(np.abs(p_np - p_tf).max())
same_class = bool((p_np.argmax(1) == p_tf.argmax(1)).all())
print(f"Images compared: {len(x)} | max probability difference: {max_diff:.2e} | same class for all: {same_class}")
sys.exit(0 if (same_class and max_diff < 1e-4) else 1)
