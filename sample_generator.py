"""
Traffic Light Recognition System using Deep Learning and IoT
Module: sample_generator.py
Description:
    Generates diverse, realistic test images for quick verification in the Streamlit
    Image Upload tab (Red light, Yellow light, Green light, Night street, Road without signal).
"""

import os
import cv2
import numpy as np
from dataset_generator import create_casing_and_bulbs, create_none_background


def generate_sample_test_images(output_dir: str):
    """Generates 5 realistic sample images for testing the detection pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[SAMPLES] Generating evaluation test images in: {output_dir}")

    samples = [
        ("test_red_light.jpg", "RED", (240, 320)),
        ("test_yellow_light.jpg", "YELLOW", (240, 320)),
        ("test_green_light.jpg", "GREEN", (240, 320)),
    ]

    for fname, sig, (w, h) in samples:
        img = create_casing_and_bulbs(w, h, active_signal=sig)
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, img)
        print(f"  -> Generated: {fname}")

    # Generate realistic night intersection with traffic light in corner
    w_night, h_night = 640, 480
    night_scene = np.zeros((h_night, w_night, 3), dtype=np.uint8)
    night_scene[:] = (20, 22, 28)  # Dark night sky

    # Road surface
    cv2.fillPoly(night_scene, [np.array([[0, h_night], [w_night // 2 - 50, int(h_night * 0.55)],
                                          [w_night // 2 + 50, int(h_night * 0.55)], [w_night, h_night]])], (45, 45, 48))
    # Lane divider
    cv2.line(night_scene, (w_night // 2, int(h_night * 0.55)), (w_night // 2, h_night), (220, 220, 220), 3)

    # Overhead traffic light gantry pole
    cv2.line(night_scene, (int(w_night * 0.72), int(h_night * 0.2)), (int(w_night * 0.72), int(h_night * 0.6)), (80, 80, 85), 6)
    cv2.line(night_scene, (int(w_night * 0.55), int(h_night * 0.2)), (int(w_night * 0.72), int(h_night * 0.2)), (80, 80, 85), 5)

    # Embed traffic light with matched night sky ambient
    tl_w, tl_h = 70, 150
    tl_img = create_casing_and_bulbs(tl_w, tl_h, active_signal="RED", ambient_mode="night")
    x_pos = int(w_night * 0.52)
    y_pos = int(h_night * 0.18)
    night_scene[y_pos:y_pos+tl_h, x_pos:x_pos+tl_w] = tl_img
    cv2.imwrite(os.path.join(output_dir, "test_night_street.jpg"), night_scene)
    print("  -> Generated: test_night_street.jpg")

    # Generate road without traffic light (NONE test)
    no_sig = create_none_background(400, 300)
    cv2.imwrite(os.path.join(output_dir, "test_road_no_signal.jpg"), no_sig)
    print("  -> Generated: test_road_no_signal.jpg")

    print("[SAMPLES] All sample test images generated successfully.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(base_dir, "sample_test_images")
    generate_sample_test_images(sample_dir)
