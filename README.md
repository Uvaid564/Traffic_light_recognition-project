# 🚦 Traffic Light Recognition System Using Deep Learning and IoT

A complete, production-grade, college-level Intelligent Transportation System (ITS) combining **Deep Learning Computer Vision** with **Internet of Things (IoT)** edge computing. The system detects and classifies traffic signals (RED, YELLOW, GREEN, and NONE) in real-time from live webcam video feeds or uploaded photographs, renders driver assistance HUD alerts with bounding boxes, and synchronizes with an embedded IoT microcontroller simulation (ESP32-S3) managing hardware GPIO actuators and MQTT smart city telemetry.

---

## 📋 Table of Contents
1. [Project Overview & Objectives](#-project-overview--objectives)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Hardware & IoT Integration](#-hardware--iot-integration)
5. [Deep Learning Model Architecture](#-deep-learning-model-architecture)
6. [Mathematical Formulations](#-mathematical-formulations)
7. [Project File Structure](#-project-file-structure)
8. [Installation & Setup](#-installation--setup)
9. [Dataset Preparation & Training](#-dataset-preparation--training)
10. [Running the Application](#-running-the-application)
11. [College Viva Voce Q&A](#-college-viva-voce-qa)

---

## 🎯 Project Overview & Objectives
* **Objective:** Design and implement an automated Advanced Driver Assistance System (ADAS) and Smart City Traffic Controller that accurately identifies traffic signals under challenging lighting conditions (daylight, dusk, rain, nighttime bloom) and communicates decisions to IoT edge hardware.
* **Domain:** Computer Vision, Convolutional Neural Networks (CNN), Embedded IoT Telemetry, Smart Transportation.
* **Target Audience:** College final-year projects, academic research demonstrations, IoT hackathons, and ADAS prototypes.

---

## ✨ Key Features
- **Real-Time Video & Webcam Recognition:** Processes live camera streams at 30+ FPS using OpenCV.
- **Bounding Box & HUD Overlay:** Dynamically tracks signal heads and displays class label, confidence percentage, and automotive ADAS warning banners (`STOP MANDATORY`, `PREPARE TO STOP`, `CLEAR TO PROCEED`).
- **High-Accuracy CNN Classifier:** Custom 3-block Convolutional Neural Network trained with Batch Normalization, Dropout regularization, and Softmax output.
- **Negative Class Rejection (`NONE`):** Prevents false positives by distinguishing road clutter, trees, signs, and unlit casings from active signals.
- **Image Upload & Preset Testing:** Drag-and-drop support for JPG/PNG images plus 1-click preset sample scenes (including nighttime intersections).
- **IoT Smart City Edge Simulation:**
  - Microcontroller GPIO simulation (ESP32-S3 / Arduino) for Red, Yellow, Green LEDs, piezo audio buzzer, and servo barrier gate.
  - JSON-formatted MQTT telemetry broadcast (`smartcity/v1/ESP32-INTERSECTION-NODE-01/telemetry`).
  - **Emergency Vehicle Preemption (EVO):** Priority corridor override for ambulances and fire engines.
  - Pedestrian crosswalk pushbutton queueing.
- **Interactive Streamlit Web Dashboard:** Modern dark-mode interface with real-time video, analytics charts, and model retraining controls.

---

## 🏗️ System Architecture

```text
+-----------------------------------------------------------------------------------------+
|                                   SYSTEM ARCHITECTURE                                   |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  [ Video / Image Input ]          [ Computer Vision ROI ]          [ Deep Learning CNN ]|
|  - Live Webcam Stream     --->    - Multi-spectrum HSV      --->   - 3 Conv2D Blocks    |
|  - Dashcam Uploads                - Morphological Dilation         - Global Avg Pooling |
|                                   - Aspect Ratio Filter            - Softmax (4 Classes)|
|                                                                             |           |
|                                                                             v           |
|                                                                   [ Signal Classification|
|                                                                   RED / YELLOW / GREEN  |
|                                                                             |           |
|                         +---------------------------------------------------+           |
|                         |                                                               |
|                         v                                                               v
|            [ Streamlit Dashboard ]                                            [ IoT Edge Node ]
|            - HUD Bounding Boxes                                               - ESP32-S3 Controller
|            - Confidence Scores                                                - GPIO Actuators
|            - Driver Alerts                                                    - MQTT Telemetry Stream
|            - Performance Curves                                               - Emergency Preemption
+-----------------------------------------------------------------------------------------+
```

---

## 🔌 Hardware & IoT Integration

### Microcontroller Pinout (ESP32-S3)
| Component | Device Pin | Logic Level | Function |
| :--- | :--- | :--- | :--- |
| **Red LED** | GPIO 21 | Digital OUT (3.3V) | High when RED signal active |
| **Yellow LED** | GPIO 19 | Digital OUT (3.3V) | High when YELLOW signal active |
| **Green LED** | GPIO 18 | Digital OUT (3.3V) | High when GREEN signal active |
| **Piezo Buzzer** | GPIO 5 | PWM Audio OUT | 1200 Hz pedestrian chirp / 600 Hz warning |
| **Servo Motor** | GPIO 4 | PWM Control OUT | 0° (Barrier Open) to 90° (Barrier Closed) |
| **LDR Sensor** | GPIO 34 | Analog ADC IN | Ambient lux measurement |
| **Push Button**| GPIO 14 | Digital IN (Pullup)| Pedestrian crosswalk request |

### Sample MQTT Telemetry Packet
```json
{
  "mqtt_topic": "smartcity/v1/ESP32-INTERSECTION-NODE-01/telemetry",
  "timestamp": "2026-11-09 13:30:00",
  "node_metadata": {
    "device_id": "ESP32-INTERSECTION-NODE-01",
    "hardware": "ESP32-S3 Dual Core 240MHz",
    "ping_ms": 22
  },
  "traffic_control": {
    "active_signal": "RED",
    "confidence_pct": 98.42,
    "action_command": "STOP - Vehicle halt mandatory; crosswalk accessible",
    "emergency_override": false
  },
  "hardware_actuators": {
    "gpio_pins": {
      "red_led": {"pin": 21, "state": "HIGH"},
      "yellow_led": {"pin": 19, "state": "LOW"},
      "green_led": {"pin": 18, "state": "LOW"},
      "servo_barrier": {"pin": 4, "angle_deg": 90, "status": "CLOSED"}
    }
  }
}
```

---

## 🧠 Deep Learning Model Architecture

The classification core is an optimized Deep Convolutional Neural Network (CNN) with Global Average Pooling:
- **Input Dimension:** `64 x 64 x 3` (RGB normalized to `[0.0, 1.0]`)
- **Block 1:** `Conv2D(32, 3x3, ReLU, padding='same') -> MaxPooling2D(2x2)`
- **Block 2:** `Conv2D(64, 3x3, ReLU, padding='same') -> MaxPooling2D(2x2)`
- **Block 3:** `Conv2D(64, 3x3, ReLU, padding='same')`
- **Spatial Pooling:** `GlobalAveragePooling2D()` (replaces Flatten to eliminate overfitting and drop parameter count from 1,000,000+ to 60,740)
- **Classifier Head:** `Dense(64, ReLU) -> Dropout(0.30) -> Dense(4, Softmax)`
- **Total Trainable Parameters:** `60,740` (237 KB)

---

## 📐 Mathematical Formulations

1. **2D Convolution:**
   $$S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i - m, j - n) K(m, n)$$
2. **Global Average Pooling (GAP):**
   $$GAP_k = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W f_k(i, j)$$
3. **Softmax Output Function:**
   $$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}} \quad \text{for } i \in \{1, \dots, K\}$$
4. **Categorical Cross-Entropy Loss:**
   $$\mathcal{L} = -\sum_{c=1}^M y_{o, c} \log(p_{o, c})$$

---

## 📁 Project File Structure

```text
traffic_light_recognition/
│
├── .streamlit/                 # Streamlit configuration
│   └── config.toml             # Headless and browser settings
│
├── dataset/                    # Training and validation dataset directory
│   ├── red/                    # Red signal images
│   ├── yellow/                 # Yellow signal images
│   ├── green/                  # Green signal images
│   └── none/                   # Negative/background scenes
│
├── models/                     # Saved models and training metrics
│   ├── traffic_light_cnn.keras # Trained CNN weights (100% Val Accuracy)
│   ├── model_metrics.json      # Training metadata & accuracy records
│   ├── training_performance.png# Accuracy & loss curves plot
│   └── confusion_matrix.png    # Validation confusion matrix heatmap
│
├── sample_test_images/         # Ready-to-use sample evaluation images
│   ├── test_red_light.jpg
│   ├── test_yellow_light.jpg
│   ├── test_green_light.jpg
│   ├── test_night_street.jpg
│   └── test_road_no_signal.jpg
│
├── model.py                    # CNN architecture definition, save/load, prediction routines
├── detection.py                # Computer vision ROI extraction, Optical Color Saliency & HUD
├── iot_simulation.py           # IoT node state machine, GPIO states & MQTT generator
├── dataset_generator.py        # Turnkey synthetic dataset generator
├── sample_generator.py         # Sample test image generator
├── train.py                    # End-to-end model training pipeline
├── test_system.py              # Automated test suite and inference validation
├── main.py                     # Professional Streamlit web dashboard
├── run_dashboard.bat           # 1-click Windows 11 launcher (Batch)
├── run_dashboard.ps1           # 1-click Windows 11 launcher (PowerShell)
├── requirements.txt            # Web app dependencies (Render)
├── requirements-train.txt      # + TensorFlow for training
├── render.yaml                 # Render deployment blueprint
└── README.md                   # Full college project documentation & viva guide
```

---

## ⚙️ Installation & Setup

### 1. Clone or Open Workspace
Navigate to the project root directory:
```bash
cd Traffic-Light-Recognition
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt          # run the dashboard only
pip install -r requirements-train.txt    # + TensorFlow, to retrain the model
```

### 3. Deploy on Render (free plan)
The web app does **not** need TensorFlow: `model.py` runs the trained CNN with a
lightweight NumPy engine that reads `models/traffic_light_cnn.keras` directly
(same output as Keras; check with `python test_numpy_engine.py`). This keeps RAM near
200 MB, inside Render's 512 MB free limit.

1. Push this folder to GitHub (with `models/` and `sample_test_images/` as folders, not zips).
2. Render → **New → Blueprint** → select the repo (uses `render.yaml`), **or** New → Web Service with:
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run main.py --server.port $PORT --server.address 0.0.0.0`
   - Environment variable: `PYTHON_VERSION = 3.12.8`

---

## 🏋️ Dataset Preparation & Training

You can train the model from the terminal in one command. If the dataset is not yet generated, `train.py` automatically generates a balanced 600-image dataset:

```bash
python train.py --epochs 15 --batch-size 32 --samples 150
```

### Generated Artifacts:
- `models/traffic_light_cnn.keras`: Model weights.
- `models/training_performance.png`: Loss and accuracy curves.
- `models/confusion_matrix.png`: Annotated confusion matrix heatmap.
- `models/model_metrics.json`: Accuracy numbers.

---

## 🚀 Running the Application

Launch the Streamlit dashboard:
```bash
streamlit run main.py
```

Open your browser at `http://localhost:8501`.

### Dashboard Navigation:
1. **📤 Image Upload & Analysis:** Upload test images or choose from presets. View original vs annotated output, class probabilities, and IoT actuator updates.
2. **📹 Live Webcam Vision:** Test in real-time with your webcam or capture snapshots.
3. **🌐 IoT Smart Traffic Simulator:** Interactive intersection panel with animated signal bulbs, manual signal overrides, emergency vehicle priority trigger, and MQTT JSON stream.
4. **📊 Model Analytics & Training:** View layer-by-layer CNN architecture, confusion matrix, loss curves, and trigger retraining.
5. **📖 College Project Report & Viva:** Read architectural theory, formulas, and 15+ viva exam questions with answers.

---

## 🎓 College Viva Voce Q&A

**Q1: What is the main objective of this project?**  
*A:* To build an intelligent, vision-based traffic light recognition system using Deep Learning (CNN) that alerts drivers/autonomous vehicles and communicates signal status to IoT smart city infrastructure for adaptive traffic control.

**Q2: Why use a CNN rather than classical color thresholding alone?**  
*A:* Color thresholding fails under varying sunlight, lens glare, neon signs, and nighttime reflections. A CNN learns spatial shapes (circular bulbs, visors, rectangular casing) combined with spectral features, providing robustness against false positives.

**Q3: Why is the 'NONE' class essential?**  
*A:* A 3-class model (Red/Yellow/Green) forced through Softmax will always assign 100% of the probability across those 3 classes, incorrectly classifying roads, cars, or trees as signals. Adding a 'NONE' class ensures the model remains silent when no traffic light is visible.

**Q4: How does the IoT simulation represent a real-world edge node?**  
*A:* It simulates an ESP32-S3 microcontroller managing physical GPIO pins (LED lights, barrier servos, buzzers), reads simulated environmental sensors (lux, vehicle queues), and transmits MQTT JSON telemetry packets to a smart city traffic management center.

**Q5: What is Emergency Vehicle Preemption (EVO)?**  
*A:* An intelligent override protocol that immediately grants a green light corridor to arriving ambulances or fire engines while holding intersecting traffic with red lights, drastically reducing emergency response times.
