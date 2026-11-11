"""
Traffic Light Recognition System using Deep Learning and IoT
Module: detection.py
Description:
    Real-time Computer Vision and Deep Learning detection pipeline.
    Combines HSV color-saliency candidate ROI localization with CNN inference,
    bounding box rendering, confidence scoring, ADAS action alerts, and IoT sync.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from model import predict_signal, CLASSES, IMG_SIZE


class TrafficLightDetector:
    """
    Intelligent detector integrating computer vision candidate localization
    with Deep Learning CNN classification and Optical Color Fusion.
    """
    def __init__(self, model=None, min_confidence: float = 0.50):
        self.model = model
        self.min_confidence = min_confidence

    def set_model(self, model):
        """Update active CNN model."""
        self.model = model

    @staticmethod
    def analyze_color_salience(roi_bgr: np.ndarray) -> Dict[str, float]:
        """Calculates optical color energy across traffic signal spectral bands with intensity thresholding."""
        if roi_bgr is None or roi_bgr.size == 0:
            return {"RED": 0.0, "YELLOW": 0.0, "GREEN": 0.0, "NONE": 100.0}

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        h_roi, w_roi = roi_bgr.shape[:2]

        # High-intensity glowing bulb thresholds (avoids false positives on trees or sky)
        m_red1 = cv2.inRange(hsv, np.array([0, 90, 150]), np.array([10, 255, 255]))
        m_red2 = cv2.inRange(hsv, np.array([165, 90, 150]), np.array([180, 255, 255]))
        m_red = cv2.bitwise_or(m_red1, m_red2)
        m_yel = cv2.inRange(hsv, np.array([15, 90, 160]), np.array([38, 255, 255]))
        m_grn = cv2.inRange(hsv, np.array([42, 90, 150]), np.array([90, 255, 255]))

        r_cnt = cv2.countNonZero(m_red)
        y_cnt = cv2.countNonZero(m_yel)
        g_cnt = cv2.countNonZero(m_grn)
        tot = r_cnt + y_cnt + g_cnt

        min_px = max(60, int(h_roi * w_roi * 0.005))
        if tot < min_px:
            return {"RED": 0.0, "YELLOW": 0.0, "GREEN": 0.0, "NONE": 100.0}

        return {
            "RED": round((r_cnt / tot) * 100.0, 1),
            "YELLOW": round((y_cnt / tot) * 100.0, 1),
            "GREEN": round((g_cnt / tot) * 100.0, 1),
            "NONE": 0.0
        }

    def classify_roi(self, roi: np.ndarray) -> Dict[str, Any]:
        """Fuses CNN probabilities with HSV optical colour energy to refine the prediction."""
        cnn = predict_signal(self.model, roi)

        # If CNN confidently identifies negative background (NONE), respect it!
        if cnn["label"] == "NONE" and cnn["confidence"] >= 65.0:
            return cnn

        color = self.analyze_color_salience(roi)

        # Only fuse if optical energy is clearly active and CNN sees traffic light features
        if color["NONE"] < 50.0 and cnn["probabilities"].get("NONE", 0.0) < 75.0:
            fused_probs = {}
            for c in CLASSES:
                if c == "NONE":
                    fused_probs[c] = round(cnn["probabilities"]["NONE"] * 0.15, 1)
                else:
                    # 45% CNN + 55% Optical Color Energy
                    fused_probs[c] = round(0.45 * cnn["probabilities"][c] + 0.55 * color[c], 1)

            best_label = max(["RED", "YELLOW", "GREEN"], key=lambda k: fused_probs[k])
            best_conf = fused_probs[best_label]

            action_alerts = {
                "RED": "STOP! Red signal active. Vehicle halt is mandatory.",
                "YELLOW": "CAUTION! Yellow signal active. Decelerate and prepare to stop.",
                "GREEN": "GO! Green signal active. Clear to proceed through intersection.",
                "NONE": "NO SIGNAL: No active traffic light identified."
            }

            return {
                "label": best_label,
                "confidence": best_conf,
                "probabilities": fused_probs,
                "action_alert": action_alerts.get(best_label, cnn["action_alert"])
            }
        return cnn

    def detect_candidate_regions(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Locates candidate traffic light bounding boxes using multi-spectrum color masks.
        """
        h_frame, w_frame = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Multi-color thresholding for active signal lights
        mask_red1 = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([165, 70, 70]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_yellow = cv2.inRange(hsv, np.array([15, 70, 70]), np.array([38, 255, 255]))
        mask_green = cv2.inRange(hsv, np.array([40, 70, 70]), np.array([92, 255, 255]))

        combined_mask = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_yellow, mask_green))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        processed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        processed_mask = cv2.dilate(processed_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes = []

        min_area = (w_frame * h_frame) * 0.0006
        max_area = (w_frame * h_frame) * 0.85

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = float(h) / max(1, w)

                if aspect >= 1.3:
                    # Already captures vertical traffic light housing
                    pad_x = int(w * 0.12)
                    pad_y = int(h * 0.08)
                else:
                    # Single bulb, expand vertically
                    pad_x = int(w * 0.35)
                    pad_y = int(h * 0.85)

                nx = max(0, x - pad_x)
                ny = max(0, y - pad_y)
                nw = min(w_frame - nx, w + 2 * pad_x)
                nh = min(h_frame - ny, h + 2 * pad_y)

                if 0.7 <= (float(nh) / max(1, nw)) <= 5.0:
                    candidate_boxes.append((nx, ny, nw, nh))

        candidate_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        return candidate_boxes[:5]

    def process_frame(self, frame_bgr: np.ndarray, iot_node=None) -> Dict[str, Any]:
        """
        Full detection pipeline on an image or video frame.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return {"error": "Invalid frame", "annotated_frame": None}

        annotated_frame = frame_bgr.copy()
        h_frame, w_frame = frame_bgr.shape[:2]
        
        detections = []
        best_detection = {
            "label": "NONE",
            "confidence": 0.0,
            "bbox": None,
            "action_alert": "NO SIGNAL: Searching for traffic signals...",
            "probabilities": {cls_name: 0.0 for cls_name in CLASSES}
        }

        # 1. Search for candidate ROIs
        candidate_boxes = self.detect_candidate_regions(frame_bgr)

        # If candidates found, classify each using CNN + Optical Saliency
        if self.model is not None and candidate_boxes:
            for (bx, by, bw, bh) in candidate_boxes:
                roi = frame_bgr[by:by+bh, bx:bx+bw]
                if roi.size == 0 or roi.shape[0] < 8 or roi.shape[1] < 8:
                    continue

                pred = self.classify_roi(roi)
                if pred["label"] != "NONE" and (pred["confidence"] / 100.0) >= self.min_confidence:
                    detection_item = {
                        "label": pred["label"],
                        "confidence": pred["confidence"],
                        "bbox": (bx, by, bw, bh),
                        "action_alert": pred["action_alert"],
                        "probabilities": pred["probabilities"]
                    }
                    detections.append(detection_item)
                    if detection_item["confidence"] > best_detection["confidence"]:
                        best_detection = detection_item

        # 2. If no candidate detected or ROI failed, evaluate full frame
        if best_detection["label"] == "NONE" and self.model is not None:
            full_pred = self.classify_roi(frame_bgr)
            if full_pred["label"] != "NONE" and (full_pred["confidence"] / 100.0) >= self.min_confidence:
                best_detection = {
                    "label": full_pred["label"],
                    "confidence": full_pred["confidence"],
                    "bbox": (int(w_frame * 0.15), int(h_frame * 0.05), int(w_frame * 0.70), int(h_frame * 0.90)),
                    "action_alert": full_pred["action_alert"],
                    "probabilities": full_pred["probabilities"]
                }
                detections.append(best_detection)
            elif full_pred["label"] == "NONE":
                best_detection["probabilities"] = full_pred["probabilities"]

        # 3. Synchronize with IoT simulation node if provided
        iot_telemetry = None
        if iot_node is not None:
            iot_telemetry = iot_node.sync_with_ai_recognition(
                best_detection["label"],
                best_detection["confidence"]
            )

        # 4. Render Bounding Boxes & HUD overlay
        annotated_frame = self._render_hud_and_boxes(
            annotated_frame,
            detections,
            best_detection,
            iot_telemetry
        )

        return {
            "annotated_frame": annotated_frame,
            "best_detection": best_detection,
            "all_detections": detections,
            "iot_telemetry": iot_telemetry
        }

    def _render_hud_and_boxes(self, frame: np.ndarray, detections: List[Dict[str, Any]],
                              best: Dict[str, Any], iot_telemetry: Optional[Dict[str, Any]]) -> np.ndarray:
        """Draws aesthetic bounding boxes and modern automotive HUD overlay."""
        h, w = frame.shape[:2]
        palette = {
            "RED": (30, 30, 240),
            "YELLOW": (20, 215, 255),
            "GREEN": (50, 220, 60),
            "NONE": (130, 130, 130)
        }

        for det in detections:
            bbox = det.get("bbox")
            if bbox is not None:
                bx, by, bw, bh = bbox
                label = det["label"]
                conf = det["confidence"]
                box_color = palette.get(label, (200, 200, 200))

                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), box_color, 2)
                tick_len = min(15, bw // 4, bh // 4)
                cv2.line(frame, (bx, by), (bx + tick_len, by), box_color, 4)
                cv2.line(frame, (bx, by), (bx, by + tick_len), box_color, 4)
                cv2.line(frame, (bx + bw, by + bh), (bx + bw - tick_len, by + bh), box_color, 4)
                cv2.line(frame, (bx + bw, by + bh), (bx + bw, by + bh - tick_len), box_color, 4)

                tag_text = f"{label} {conf:.1f}%"
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.55
                thickness = 1
                (tw, th), _ = cv2.getTextSize(tag_text, font, font_scale, thickness)
                cv2.rectangle(frame, (bx, max(0, by - th - 10)), (bx + tw + 12, max(0, by)), box_color, -1)
                cv2.putText(frame, tag_text, (bx + 6, max(th + 2, by - 4)), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # HUD Top Banner Overlay
        narrow = w < 500            # small images: avoid text running off the right edge
        hud_h = 90 if narrow else 75
        hud_overlay = frame.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (15, 18, 22), -1)
        frame = cv2.addWeighted(hud_overlay, 0.78, frame, 0.22, 0)

        signal_color = palette.get(best["label"], (100, 100, 100))
        cv2.line(frame, (0, hud_h), (w, hud_h), signal_color, 2)

        cv2.putText(frame, "TRAFFIC AI VISION | DEEP LEARNING + IoT", (15, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 215, 230), 1, cv2.LINE_AA)

        status_text = f"SIGNAL: {best['label']} ({best['confidence']:.1f}%)" if best['label'] != 'NONE' else "SIGNAL: NONE (CLEAR ROAD)"
        cv2.putText(frame, status_text, (15, 54),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, signal_color, 2, cv2.LINE_AA)

        action_text = best["action_alert"]
        action_pos = (15, 70) if narrow else (int(w * 0.45), 32)
        action_scale = 0.38 if narrow else 0.48
        cv2.putText(frame, action_text, action_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, action_scale, (240, 240, 240), 1, cv2.LINE_AA)

        if iot_telemetry:
            pins = iot_telemetry.get("hardware_actuators", {}).get("gpio_pins", {})
            gate_status = pins.get("servo_barrier", {}).get("status", "OPEN")
            iot_badge = f"IoT: {iot_telemetry.get('node_metadata', {}).get('device_id', 'ESP32')} | GATE: {gate_status} | LATENCY: {iot_telemetry.get('node_metadata', {}).get('ping_ms', 24)}ms"
            badge_pos = (15, 85) if narrow else (int(w * 0.45), 58)
            badge_scale = 0.36 if narrow else 0.42
            cv2.putText(frame, iot_badge, badge_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, badge_scale, (100, 225, 200), 1, cv2.LINE_AA)

        return frame


if __name__ == "__main__":
    import os
    import sys
    from model import load_traffic_model
    from iot_simulation import default_iot_node

    print("=" * 65)
    print(" TRAFFIC LIGHT DETECTION ENGINE (CV + DEEP LEARNING)")
    print("=" * 65)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "traffic_light_cnn.keras")
    samples_dir = os.path.join(base_dir, "sample_test_images")

    if not os.path.exists(model_path):
        print(f"[!] Model file not found at: {model_path}")
        print("Please train the model first by running train.py")
        input("\nPress Enter to exit...")
        sys.exit(1)

    print("\n[1] Loading Trained Deep Learning Model...")
    model = load_traffic_model(model_path)
    detector = TrafficLightDetector(model=model, min_confidence=0.50)
    print("    -> Detector initialized with 50% confidence threshold.")

    print("\n[2] Testing Detection Pipeline on Sample Images:")
    test_files = [
        "test_red_light.jpg",
        "test_yellow_light.jpg",
        "test_green_light.jpg",
        "test_night_street.jpg",
        "test_road_no_signal.jpg"
    ]

    tested = 0
    for fname in test_files:
        fpath = os.path.join(samples_dir, fname)
        if not os.path.exists(fpath):
            print(f"    [SKIP] Sample not found: {fname}")
            continue

        img = cv2.imread(fpath)
        if img is None:
            print(f"    [SKIP] Could not decode image: {fname}")
            continue
        result = detector.process_frame(img, iot_node=default_iot_node)
        best = result["best_detection"]
        telemetry = result["iot_telemetry"] or {}
        tested += 1

        sig = best["label"]
        conf = best["confidence"]
        gate = (telemetry.get("hardware_actuators", {})
                         .get("gpio_pins", {})
                         .get("servo_barrier", {})
                         .get("status", "N/A"))

        print(f"    • {fname:<24} -> Signal: {sig:<7} ({conf:5.1f}%) | Barrier Gate: {gate:<6}")
        print(f"      Action: {best['action_alert']}")

    print("\n" + "=" * 65)
    if tested:
        print(f" DONE: detection pipeline ran on {tested} sample image(s).")
    else:
        print(" No sample images found. Run sample_generator.py first.")
    print("=" * 65)
    input("\nPress Enter to exit...")

