"""
Traffic Light Recognition System using Deep Learning and IoT
Module: train.py
Description:
    Complete end-to-end training pipeline:
    1. Prepares and validates dataset (auto-generates synthetic samples if needed).
    2. Performs data augmentation and train/validation split.
    3. Trains the CNN model with callbacks (EarlyStopping, ReduceLROnPlateau).
    4. Computes accuracy, loss, and confusion matrix.
    5. Saves the trained model weights and visualization graphs.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Headless backend for server/command line plotting
import matplotlib.pyplot as plt

from model import build_traffic_light_cnn, save_traffic_model, CLASSES, CLASS_TO_IDX, IMG_SIZE
from dataset_generator import generate_dataset


def load_dataset(dataset_dir: str, target_size=IMG_SIZE):
    """
    Loads all images and one-hot labels from subdirectories (red, yellow, green, none).
    """
    images = []
    labels = []
    
    for cls_name in CLASSES:
        folder = os.path.join(dataset_dir, cls_name.lower())
        if not os.path.isdir(folder):
            print(f"[DATA][WARN] Folder missing for class '{cls_name}': {folder}")
            continue
        
        cls_idx = CLASS_TO_IDX[cls_name]
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"[DATA] Loading '{cls_name}' ({len(files)} samples)...")
        
        for fname in files:
            fpath = os.path.join(folder, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            # Resize and normalize
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(cls_idx)

    if not images:
        raise ValueError(f"No valid images found in {dataset_dir}.")

    X = np.array(images, dtype=np.float32)
    y_raw = np.array(labels, dtype=np.int32)

    missing = [CLASSES[i] for i in range(len(CLASSES)) if not np.any(y_raw == i)]
    if missing:
        raise ValueError(f"No training images for class(es): {missing}. "
                         f"Delete the dataset folder and run train.py again to regenerate it.")
    
    # One-hot encode
    num_classes = len(CLASSES)
    y = np.zeros((len(y_raw), num_classes), dtype=np.float32)
    for i, label_idx in enumerate(y_raw):
        y[i, label_idx] = 1.0

    return X, y, y_raw


def train_test_split_data(X, y, test_ratio=0.20, seed=42):
    """Stratified split: every class keeps the same train/validation proportion."""
    rng = np.random.RandomState(seed)
    labels = np.argmax(y, axis=1)
    train_idx, val_idx = [], []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * test_ratio)))
        val_idx.extend(cls_idx[:n_val])
        train_idx.extend(cls_idx[n_val:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def augment_training_set(X, y, seed=42):
    """
    Adds augmented copies (horizontal flip + brightness jitter) of the TRAINING
    images only, so the validation set stays untouched.
    """
    rng = np.random.RandomState(seed)
    flipped = X[:, :, ::-1, :]
    factors = rng.uniform(0.75, 1.25, size=(len(X), 1, 1, 1)).astype(np.float32)
    bright = np.clip(X * factors, 0.0, 1.0).astype(np.float32)

    X_aug = np.concatenate([X, flipped, bright], axis=0)
    y_aug = np.concatenate([y, y, y], axis=0)
    order = rng.permutation(len(X_aug))
    return X_aug[order], y_aug[order]


def plot_and_save_training_curves(history, output_path: str):
    """Generates dual-panel plot for Accuracy and Loss curves."""
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-o', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_acc, 'g--s', label='Validation Accuracy', linewidth=2)
    plt.title('CNN Training & Validation Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.ylim([min(0.5, min(acc + val_acc) - 0.05), 1.02])

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'r-o', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'm--s', label='Validation Loss', linewidth=2)
    plt.title('CNN Training & Validation Loss (Cross-Entropy)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[PLOT] Training curves saved to: {output_path}")


def plot_and_save_confusion_matrix(cm, classes, output_path: str):
    """Draws annotated Confusion Matrix heatmap."""
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix: Traffic Signal Classification', fontsize=12, fontweight='bold')
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    # Annotate counts inside matrix cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, format(val, 'd'),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color=color, fontsize=12, fontweight='bold')

    plt.ylabel('True Class', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[PLOT] Confusion matrix saved to: {output_path}")


def compute_confusion_matrix(y_true, y_pred, num_classes):
    """Compute pure NumPy confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def run_training_pipeline(dataset_dir: str, models_dir: str, epochs: int = 15, batch_size: int = 32, samples_per_class: int = 150):
    """Executes complete training, evaluation, and artifact generation."""
    print("=" * 65)
    print("TRAFFIC LIGHT RECOGNITION SYSTEM - CNN TRAINING PIPELINE")
    print("=" * 65)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    # 1. Check dataset availability
    total_samples = 0
    for c in CLASSES:
        folder = os.path.join(dataset_dir, c.lower())
        if os.path.exists(folder):
            total_samples += len(os.listdir(folder))

    if total_samples < (samples_per_class * len(CLASSES) * 0.8):
        print(f"[DATASET] Insufficient data detected ({total_samples} images). Generating synthetic dataset...")
        generate_dataset(dataset_dir, num_samples_per_class=samples_per_class)

    # 2. Load dataset
    X, y, y_raw = load_dataset(dataset_dir)
    print(f"[DATA] Successfully loaded {len(X)} total images across {len(CLASSES)} classes.")

    # 3. Train / Validation Split
    X_train, y_train, X_val, y_val = train_test_split_data(X, y, test_ratio=0.20)
    X_train, y_train = augment_training_set(X_train, y_train)
    print(f"[DATA] Training set (after augmentation): {len(X_train)} samples | Validation set: {len(X_val)} samples")

    # 4. Build Model
    print("[CNN] Instantiating Deep Convolutional Neural Network...")
    model = build_traffic_light_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=len(CLASSES))
    model.summary()

    # Callbacks
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
        ]
    except Exception:
        callbacks = []

    # 5. Fit Model
    print(f"\n[TRAIN] Commencing training for {epochs} epochs with batch size {batch_size}...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    elapsed = time.time() - start_time
    print(f"[TRAIN] Training finished in {elapsed:.2f} seconds.")

    # 6. Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n[EVALUATION] Final Validation Loss:     {val_loss:.4f}")
    print(f"[EVALUATION] Final Validation Accuracy: {val_acc * 100:.2f}%")

    # Confusion matrix
    val_preds = model.predict(X_val, verbose=0)
    y_true_indices = np.argmax(y_val, axis=1)
    y_pred_indices = np.argmax(val_preds, axis=1)
    cm = compute_confusion_matrix(y_true_indices, y_pred_indices, len(CLASSES))

    # 7. Save Artifacts
    model_save_path = os.path.join(models_dir, "traffic_light_cnn.keras")
    save_traffic_model(model, model_save_path)

    perf_plot_path = os.path.join(models_dir, "training_performance.png")
    plot_and_save_training_curves(history, perf_plot_path)

    cm_plot_path = os.path.join(models_dir, "confusion_matrix.png")
    plot_and_save_confusion_matrix(cm, CLASSES, cm_plot_path)

    # Save Metrics JSON
    metrics_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs_trained": len(history.history.get('accuracy', [])),
        "final_train_accuracy": float(history.history['accuracy'][-1]),
        "final_train_loss": float(history.history['loss'][-1]),
        "final_val_accuracy": float(val_acc),
        "final_val_loss": float(val_loss),
        "classes": CLASSES,
        "input_shape": list(IMG_SIZE) + [3],
        "training_duration_seconds": round(elapsed, 2)
    }
    metrics_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[METRICS] Saved training metadata to: {metrics_path}")

    print("=" * 65)
    print("SUCCESS: Traffic Light Model Trained and Artifacts Generated!")
    print("=" * 65)
    return model, metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN for Traffic Light Recognition")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--samples", type=int, default=150, help="Samples per class")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    d_dir = os.path.join(base_dir, "dataset")
    m_dir = os.path.join(base_dir, "models")
    
    run_training_pipeline(d_dir, m_dir, epochs=args.epochs, batch_size=args.batch_size, samples_per_class=args.samples)
