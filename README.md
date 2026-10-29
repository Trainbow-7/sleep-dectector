# WakeWatch — Live Camera Sleep & Drowsiness Monitor

WakeWatch is a real-time computer vision and driver fatigue monitoring application built with **Streamlit**, **streamlit-webrtc**, and **MediaPipe**. It streams continuous webcam video via WebRTC, validates the user's face position and orientation, and detects micro-sleep, drowsiness, and yawning in real time.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
streamlit run app.py
```
Open your browser and navigate to:
```
http://localhost:8501
```

Click **START** on the webcam stream and grant camera permissions when prompted by your browser.

---

## 🎯 Key Features

1. **Continuous Real-Time WebRTC Video Stream**
   - No snapshots required (`st.camera_input()` replaced with `streamlit-webrtc`).
   - Frame-by-frame processing running asynchronously via `VideoProcessorBase`.
   - Low-latency stream powered by Google's public STUN servers.

2. **Face Capture Validation with Color Indicators**
   - **🟩 Green Bounding Box ("Face Captured"):**
     Displayed when the face meets all quality and orientation standards.
   - **🟥 Red Bounding Box ("Adjust Position"):**
     Displayed with diagnostic hints when position needs adjustment.
   - **Validation Criteria:**
     - **Single Primary Face:** Locks onto the largest/closest face and ignores background persons.
     - **Frame Boundary Margins:** Face bounding box must not touch or be cut off by frame edges (minimum 18px margin).
     - **Face Size:** Must occupy between 15% and 85% of frame width (prevents being too far or too close).
     - **Frontal Orientation:** Head yaw and pitch checked to ensure the user is facing the camera with both eyes visible.
     - **On-Screen Diagnostics:** Helpful hints such as *"Move closer"*, *"Move back"*, *"Face the camera"*, or *"Face cut off"*.
     - **Auto-Pause:** When the face indicator is RED, drowsiness detection is safely paused and timers are reset to prevent false alarms.

3. **Fatigue & Sleep Detection Engine**
   - **Eye Aspect Ratio (EAR):** Computes precise 6-point ocular geometry for left and right eyes, smoothed with a moving average filter.
   - **Mouth Aspect Ratio (MAR):** Detects yawning and prolonged mouth opening.
   - **Time-Based State Machine:**
     - 🟢 **AWAKE:** Smoothed EAR is above the threshold.
     - 🟡 **DROWSY:** Eyes closed for 1.2s – 2.9s or prolonged yawning.
     - 🔴 **SLEEPING:** Eyes closed continuously for 3.0+ seconds (triggers full alarm).

4. **Audio & Visual Alerts**
   - High-visibility flashing red overlay banner on the video stream during SLEEPING state.
   - Audible high-frequency pulsing wake-up buzzer synthesized in the browser via the Web Audio API.

5. **Thread-Safe Live Telemetry**
   - Thread-safe metric synchronization (`SharedMetrics` with `threading.Lock`) between background WebRTC video frames and the Streamlit frontend.
   - Real-time display of current EAR, eye closure seconds, FPS, and face validation diagnostics.

---

## 🛠️ Threshold Tuning Guide

All detection parameters can be adjusted in real time from the sidebar sliders:

| Parameter | Default | Recommended Range | Description |
|-----------|---------|-------------------|-------------|
| **EAR Threshold** | `0.23` | `0.20 – 0.26` | Eye Aspect Ratio threshold. Values below this trigger eye-closure tracking. Awake EAR is typically `0.28 – 0.34`. If natural blinks count as sleep, lower this (e.g. to `0.21`). If closed eyes are missed, raise it (e.g. to `0.24`). |
| **Drowsy Duration** | `1.2s` | `1.0s – 1.5s` | Seconds of continuous eye closure required to trigger a `DROWSY` warning. Standard human blinks last 0.15s – 0.35s, so values above 1.0s eliminate false blink triggers. |
| **Sleeping Duration** | `3.0s` | `2.5s – 4.0s` | Seconds of continuous eye closure required to trigger the full `SLEEPING` alert buzzer. |
| **Enable Audio Alarm** | `True` | `True / False` | Toggles the browser synthesizer wake-up buzzer. |
| **Yawn Detection (MAR)** | `True` | `True / False` | Detects yawning when Mouth Aspect Ratio exceeds `0.65`. |

---

## 📁 Project Architecture

```
sleep-dectector/
├── app.py                      # Main Streamlit + WebRTC real-time application
│   ├── FaceValidator           # Module 1: Validates face size, position, and pose
│   ├── OcularAnalyzer          # Module 2: Computes EAR, MAR, and smoothed averages
│   ├── DrowsinessStateMachine  # Module 3: Time-based AWAKE/DROWSY/SLEEPING logic
│   ├── OverlayRenderer         # Module 4: Draws bounding boxes, HUD, and banners
│   ├── SharedMetrics           # Module 5: Thread-safe metrics bridge
│   └── SleepVideoProcessor     # WebRTC VideoProcessorBase implementation
├── extract_frames.py           # Frame extraction utility for WakeWatch dataset
├── prepare_data.py             # Preprocessing & Leave-One-Person-Out data split
├── requirements.txt            # Project dependencies
├── face_landmarker.task        # MediaPipe FaceLandmarker model file
└── notebooks/
    └── wakewatch_training.ipynb# MobileNetV2 3-stage training notebook
```

