"""
Traffic Light Recognition System using Deep Learning and IoT
Module: main.py  (Streamlit web dashboard)

Run locally :  streamlit run main.py
Render      :  streamlit run main.py --server.port $PORT --server.address 0.0.0.0
"""

import json
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from model import CLASSES, load_traffic_model
from detection import TrafficLightDetector
from iot_simulation import SmartTrafficIoTNode

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "traffic_light_cnn.keras")
SAMPLES_DIR = os.path.join(BASE_DIR, "sample_test_images")

SIGNAL_HEX = {"RED": "#ef4444", "YELLOW": "#facc15", "GREEN": "#22c55e", "NONE": "#9ca3af"}

st.set_page_config(page_title="Traffic Light Recognition | DL + IoT",
                   page_icon="🚦", layout="wide")


# ----------------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading CNN model...")
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_traffic_model(MODEL_PATH)


def get_iot_node() -> SmartTrafficIoTNode:
    # One IoT node per browser session (so users don't share state)
    if "iot_node" not in st.session_state:
        st.session_state.iot_node = SmartTrafficIoTNode()
    return st.session_state.iot_node


def get_detector(min_conf: float) -> TrafficLightDetector:
    return TrafficLightDetector(model=get_model(), min_confidence=min_conf)


def decode_upload(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_result(frame_bgr, result):
    best = result["best_detection"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original**")
        st.image(bgr_to_rgb(frame_bgr), width="stretch")
    with c2:
        st.markdown("**Annotated (HUD + bounding boxes)**")
        st.image(bgr_to_rgb(result["annotated_frame"]), width="stretch")

    colour = SIGNAL_HEX.get(best["label"], "#9ca3af")
    st.markdown(
        f"<div style='padding:14px;border-radius:10px;border-left:8px solid {colour};"
        f"background:rgba(255,255,255,0.04)'>"
        f"<span style='font-size:1.4rem;font-weight:700;color:{colour}'>{best['label']}</span>"
        f" &nbsp; ({best['confidence']:.1f}%)<br>{best['action_alert']}</div>",
        unsafe_allow_html=True,
    )

    probs = best.get("probabilities") or {c: 0.0 for c in CLASSES}
    df = pd.DataFrame({"Class": list(probs.keys()), "Probability (%)": list(probs.values())})
    st.bar_chart(df, x="Class", y="Probability (%)")

    if result.get("iot_telemetry"):
        with st.expander("IoT telemetry packet (MQTT JSON)"):
            st.json(result["iot_telemetry"])


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("🚦 Traffic AI Vision")
st.sidebar.caption("Deep Learning (CNN) + IoT edge simulation")
min_conf = st.sidebar.slider("Minimum confidence", 0.30, 0.95, 0.50, 0.05)
model_loaded = get_model() is not None
if model_loaded:
    st.sidebar.success("CNN model loaded")
else:
    st.sidebar.error("Model file missing: models/traffic_light_cnn.keras")

st.title("🚦 Traffic Light Recognition System")
st.caption("CNN classification of RED / YELLOW / GREEN / NONE signals with IoT smart-intersection sync")

tab_img, tab_cam, tab_iot, tab_model, tab_about = st.tabs([
    "📤 Image Upload", "📹 Webcam Snapshot", "🌐 IoT Simulator", "📊 Model Analytics", "📖 About"
])

# ----------------------------------------------------------------------------
# Tab 1: Image upload & presets
# ----------------------------------------------------------------------------
with tab_img:
    st.subheader("Upload an image or choose a preset sample")
    presets = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))) \
        if os.path.isdir(SAMPLES_DIR) else []

    col_a, col_b = st.columns([2, 1])
    with col_a:
        uploaded = st.file_uploader("Upload JPG / PNG", type=["jpg", "jpeg", "png"])
    with col_b:
        preset = st.selectbox("...or pick a sample", ["(none)"] + presets)

    frame = None
    if uploaded is not None:
        frame = decode_upload(uploaded.getvalue())
    elif preset != "(none)":
        frame = cv2.imread(os.path.join(SAMPLES_DIR, preset))

    if frame is None:
        st.info("Upload an image or select a sample to run detection.")
    elif not model_loaded:
        st.error("Model not available, cannot run detection.")
    else:
        with st.spinner("Running detection..."):
            result = get_detector(min_conf).process_frame(frame, iot_node=get_iot_node())
        show_result(frame, result)

# ----------------------------------------------------------------------------
# Tab 2: Webcam snapshot (browser camera; works on cloud hosting)
# ----------------------------------------------------------------------------
with tab_cam:
    st.subheader("Capture a frame from your webcam")
    st.caption("The browser camera is used, so this works on Render too. "
               "Live 30 FPS video needs a local run (python detection.py).")
    shot = st.camera_input("Take a picture")
    if shot is not None and model_loaded:
        cam_frame = decode_upload(shot.getvalue())
        result = get_detector(min_conf).process_frame(cam_frame, iot_node=get_iot_node())
        show_result(cam_frame, result)

# ----------------------------------------------------------------------------
# Tab 3: IoT simulator
# ----------------------------------------------------------------------------
with tab_iot:
    node = get_iot_node()
    st.subheader("ESP32-S3 smart intersection node")

    c1, c2, c3, c4 = st.columns(4)
    for col, sig in zip((c1, c2, c3), ("RED", "YELLOW", "GREEN")):
        if col.button(f"Set {sig}", width="stretch"):
            node.sync_with_ai_recognition(sig, 100.0)
    if c4.button("🚑 Emergency preemption", width="stretch"):
        node.trigger_emergency_vehicle_preemption("AMBULANCE")

    c5, c6 = st.columns(2)
    if c5.button("🚶 Pedestrian request", width="stretch"):
        node.request_pedestrian_crossing()
    if c6.button("Clear emergency / pedestrian", width="stretch"):
        node.clear_emergency_override()
        node.reset_pedestrian_crossing()

    telemetry = node.get_latest_telemetry()
    tc = telemetry["traffic_control"]
    pins = telemetry["hardware_actuators"]["gpio_pins"]

    bulbs = ""
    for key, sig in (("red_led", "RED"), ("yellow_led", "YELLOW"), ("green_led", "GREEN")):
        on = pins[key]["state"] == "HIGH"
        colour = SIGNAL_HEX[sig] if on else "#2a2a2a"
        glow = f"box-shadow:0 0 22px {colour};" if on else ""
        bulbs += (f"<div style='width:56px;height:56px;border-radius:50%;margin:8px auto;"
                  f"background:{colour};{glow}'></div>")
    left, right = st.columns([1, 3])
    with left:
        st.markdown(f"<div style='background:#111;border-radius:14px;padding:10px;width:90px'>{bulbs}</div>",
                    unsafe_allow_html=True)
    with right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Active signal", tc["active_signal"])
        m2.metric("Barrier gate", pins["servo_barrier"]["status"])
        m3.metric("Latency", f"{telemetry['node_metadata']['ping_ms']} ms")
        st.write(f"**Mode:** {tc['control_mode']}  |  **Command:** {tc['action_command']}")
        if tc["emergency_override"]:
            st.warning(f"Emergency override active for {tc['emergency_vehicle']}")

    with st.expander("Latest MQTT telemetry JSON", expanded=False):
        st.code(json.dumps(telemetry, indent=2), language="json")

# ----------------------------------------------------------------------------
# Tab 4: Model analytics
# ----------------------------------------------------------------------------
with tab_model:
    st.subheader("Training results")
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        a, b, c, d = st.columns(4)
        a.metric("Validation accuracy", f"{m['final_val_accuracy'] * 100:.2f}%")
        b.metric("Validation loss", f"{m['final_val_loss']:.4f}")
        c.metric("Train accuracy", f"{m['final_train_accuracy'] * 100:.2f}%")
        d.metric("Epochs", m["epochs_trained"])
    for img_name, caption in (("training_performance.png", "Accuracy & loss curves"),
                              ("confusion_matrix.png", "Validation confusion matrix")):
        p = os.path.join(MODELS_DIR, img_name)
        if os.path.exists(p):
            st.image(p, caption=caption, width="stretch")

    if model_loaded:
        with st.expander("Layer-by-layer architecture"):
            mdl = get_model()
            rows = [{"Layer": l.name, "Type": getattr(l, "kind", l.__class__.__name__), "Params": l.count_params()}
                    for l in mdl.layers]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.write(f"Total parameters: **{mdl.count_params():,}**")
    st.caption("Retraining is done locally: python train.py --epochs 15 --samples 150")

# ----------------------------------------------------------------------------
# Tab 5: About
# ----------------------------------------------------------------------------
with tab_about:
    readme = os.path.join(BASE_DIR, "README.md")
    if os.path.exists(readme):
        with open(readme, encoding="utf-8") as f:
            st.markdown(f.read())
