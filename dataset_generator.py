"""
Traffic Light Recognition System using Deep Learning and IoT
Module: dataset_generator.py
Description:
    Synthetic dataset generator. Draws traffic-light housings with one active bulb
    (RED / YELLOW / GREEN) and negative background scenes (NONE).
    Used by train.py (when the dataset folder is empty) and sample_generator.py.
"""

import os
import random
from typing import Optional

import cv2
import numpy as np

# BGR colours
LIT = {"RED": (40, 40, 245), "YELLOW": (30, 215, 255), "GREEN": (70, 225, 60)}
DIM = {"RED": (25, 25, 70), "YELLOW": (25, 65, 80), "GREEN": (30, 70, 30)}
ORDER = ["RED", "YELLOW", "GREEN"]


def _background(w: int, h: int, ambient_mode: str = "day") -> np.ndarray:
    if ambient_mode == "night":
        base = np.array([random.randint(15, 30), random.randint(15, 30), random.randint(18, 35)])
    else:
        base = np.array([random.randint(110, 200), random.randint(110, 200), random.randint(110, 200)])
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = base
    noise = np.random.randint(-12, 13, (h, w, 3))
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def create_casing_and_bulbs(w: int, h: int, active_signal: Optional[str] = "RED",
                            ambient_mode: str = "day") -> np.ndarray:
    """Draws a vertical 3-bulb traffic light; `active_signal` bulb is lit with glow."""
    img = _background(w, h, ambient_mode)

    cw = int(w * random.uniform(0.40, 0.55))
    ch = int(h * random.uniform(0.80, 0.92))
    cx0 = (w - cw) // 2 + random.randint(-max(1, w // 20), max(1, w // 20))
    cy0 = (h - ch) // 2
    cx0 = max(0, min(w - cw, cx0))
    shade = random.randint(20, 45)
    cv2.rectangle(img, (cx0, cy0), (cx0 + cw, cy0 + ch), (shade, shade, shade), -1)
    cv2.rectangle(img, (cx0, cy0), (cx0 + cw, cy0 + ch), (shade + 25,) * 3, 1)

    radius = max(2, int(min(cw * 0.36, ch / 7.0)))
    centre_x = cx0 + cw // 2
    active = (active_signal or "NONE").upper()

    glow = np.zeros_like(img)
    for i, colour in enumerate(ORDER):
        cy = cy0 + int(ch * (i + 0.5) / 3.0)
        if colour == active:
            cv2.circle(glow, (centre_x, cy), int(radius * 1.8), LIT[colour], -1)
            cv2.circle(img, (centre_x, cy), radius, LIT[colour], -1)
            cv2.circle(img, (centre_x - radius // 3, cy - radius // 3), max(1, radius // 3),
                       (235, 235, 255), -1)
        else:
            cv2.circle(img, (centre_x, cy), radius, DIM[colour], -1)

    if active in LIT:
        k = max(3, (radius * 2) | 1)
        glow = cv2.GaussianBlur(glow, (k, k), 0)
        img = cv2.addWeighted(img, 1.0, glow, 0.55, 0)
    return img


def create_none_background(w: int, h: int) -> np.ndarray:
    """Negative scenes: road, tiles/windows, foliage, or an unlit traffic light."""
    kind = random.choice(["road", "tiles", "foliage", "unlit"])
    if kind == "unlit":
        return create_casing_and_bulbs(w, h, active_signal="NONE",
                                       ambient_mode=random.choice(["day", "night"]))

    img = _background(w, h, "day")
    if kind == "road":
        img[:] = (random.randint(40, 70),) * 3
        cv2.line(img, (w // 2, 0), (w // 2, h), (220, 220, 220), max(1, w // 40))
        cv2.rectangle(img, (0, 0), (w, h // 3), (random.randint(150, 220), 170, 140), -1)
    elif kind == "tiles":
        img[:] = (random.randint(80, 110), random.randint(90, 115), random.randint(90, 110))
        step = max(6, w // random.randint(8, 22))
        col = (random.randint(160, 210), random.randint(170, 200), random.randint(180, 210))
        for y in range(step // 4, h, step):
            for x in range(step // 4, w, step):
                cv2.rectangle(img, (x, y), (x + step // 2, y + step // 2), col, -1)
    else:  # foliage (dull, low-brightness greens so it is not confused with a lit bulb)
        for _ in range(max(10, (w * h) // 300)):
            c = (random.randint(20, 60), random.randint(60, 110), random.randint(20, 60))
            cv2.circle(img, (random.randint(0, w), random.randint(0, h)),
                       random.randint(2, max(3, w // 10)), c, -1)
    return img


def generate_dataset(output_dir: str, num_samples_per_class: int = 150, size: int = 64):
    """Writes a balanced synthetic dataset to output_dir/{red,yellow,green,none}/."""
    print(f"[DATASET] Generating {num_samples_per_class} images per class in: {output_dir}")
    for cls in ["RED", "YELLOW", "GREEN", "NONE"]:
        folder = os.path.join(output_dir, cls.lower())
        os.makedirs(folder, exist_ok=True)
        for i in range(num_samples_per_class):
            if cls == "NONE":
                img = create_none_background(size, size)
            else:
                img = create_casing_and_bulbs(size, size, active_signal=cls,
                                              ambient_mode=random.choice(["day", "day", "night"]))
            cv2.imwrite(os.path.join(folder, f"{cls.lower()}_{i:04d}.jpg"), img)
    print("[DATASET] Done.")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    generate_dataset(os.path.join(base, "dataset"), num_samples_per_class=150)
