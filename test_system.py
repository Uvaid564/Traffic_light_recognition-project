"""
Traffic Light Recognition System using Deep Learning and IoT
Module: test_system.py
Description:
    Verification test suite to validate:
    1. Model loading
    2. Sample image inference (compared against the expected signal)
    3. IoT node state synchronization
    4. Bounding box & HUD generation
"""

import os
import sys

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
MODEL_PATH = os.path.join(BASE_DIR, "models", "traffic_light_cnn.keras")
SAMPLES_DIR = os.path.join(BASE_DIR, "sample_test_images")

from model import load_traffic_model
from detection import TrafficLightDetector
from iot_simulation import default_iot_node

# Image -> signal the system is expected to report
EXPECTED = {
    "test_red_light.jpg": "RED",
    "test_yellow_light.jpg": "YELLOW",
    "test_green_light.jpg": "GREEN",
    "test_night_street.jpg": "RED",
    "test_road_no_signal.jpg": "NONE",
}


def ensure_samples():
    """Generate the sample images automatically if they are missing."""
    missing = [f for f in EXPECTED if not os.path.exists(os.path.join(SAMPLES_DIR, f))]
    if missing:
        print(f"    Missing {len(missing)} sample image(s) - generating them now...")
        from sample_generator import generate_sample_test_images
        generate_sample_test_images(SAMPLES_DIR)


def run_tests():
    print("=" * 65)
    print("SYSTEM VERIFICATION & INFERENCE TEST SUITE")
    print("=" * 65)

    if not os.path.exists(MODEL_PATH):
        print(f"[FAIL] Model file not found at: {MODEL_PATH}")
        print("       Train the model first by running train.py")
        return False

    print("[1] Loading Trained Deep Learning Model...")
    model = load_traffic_model(MODEL_PATH)
    print("    -> Model loaded successfully.")

    print("\n[2] Initializing Computer Vision & IoT Detection Engine...")
    detector = TrafficLightDetector(model=model, min_confidence=0.50)
    print("    -> Detector initialized with confidence threshold: 50%")

    print("\n[3] Running Inference across Test Images:")
    ensure_samples()

    passed, failed = 0, 0
    for fname, expected in EXPECTED.items():
        fpath = os.path.join(SAMPLES_DIR, fname)
        img = cv2.imread(fpath)
        if img is None:
            print(f"    [FAIL] Could not read image: {fname}")
            failed += 1
            continue

        res = detector.process_frame(img, iot_node=default_iot_node)
        best = res["best_detection"]
        telemetry = res["iot_telemetry"] or {}
        gate = (telemetry.get("hardware_actuators", {})
                         .get("gpio_pins", {})
                         .get("servo_barrier", {})
                         .get("status", "N/A"))

        ok = best["label"] == expected
        passed += ok
        failed += (not ok)
        status = "PASS" if ok else "FAIL"
        print(f"    [{status}] {fname:<24} expected {expected:<7} got {best['label']:<7} "
              f"({best['confidence']:5.1f}%) | Gate: {gate}")

        if res.get("annotated_frame") is None:
            print("           -> HUD/bounding-box frame was not generated")
            failed += 1

    print("\n[4] Validating IoT Telemetry Packet...")
    try:
        telemetry = default_iot_node.get_latest_telemetry()
        pins = telemetry['hardware_actuators']['gpio_pins']
        print(f"    -> Topic:      {telemetry['mqtt_topic']}")
        print(f"    -> Controller: {telemetry['node_metadata']['device_id']}")
        print(f"    -> Latency:    {telemetry['node_metadata']['ping_ms']} ms")
        print(f"    -> Actuators:  Red={pins['red_led']['state']}, "
              f"Yellow={pins['yellow_led']['state']}, Green={pins['green_led']['state']}")
    except (AttributeError, KeyError, TypeError) as exc:
        print(f"    [FAIL] IoT telemetry packet is missing or malformed: {exc!r}")
        failed += 1

    print("\n" + "=" * 65)
    all_passed = failed == 0 and passed == len(EXPECTED)
    if all_passed:
        print(f"ALL TESTS PASSED ({passed}/{len(EXPECTED)} images correct): System Ready for Demonstration!")
    else:
        print(f"TESTS FAILED: {passed}/{len(EXPECTED)} images correct, {failed} problem(s) found.")
        print("Tip: retrain with more epochs/samples (python train.py --epochs 25 --samples 300).")
    print("=" * 65)
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
