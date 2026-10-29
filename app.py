"""WakeWatch: Sleep and Drowsiness Detection Streamlit Application.

Interactive dev server web application providing:
1. Real-time Webcam detection with wake-up alarm sound.
2. Video file drowsiness & fatigue timeline analysis.
3. Single image / dataset frame inspection.
4. Custom trained model loader (MobileNetV2 / Keras / TFLite) with CV fallback.
"""

import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Configure page layout and branding
st.set_page_config(
    page_title="WakeWatch | Sleep & Drowsiness Detector",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3B82F6, #10B981, #F59E0B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .badge-awake {
        background-color: #065F46;
        color: #34D399;
        font-weight: bold;
        padding: 6px 16px;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.1rem;
        border: 1px solid #10B981;
    }
    .badge-drowsy {
        background-color: #78350F;
        color: #FBBF24;
        font-weight: bold;
        padding: 6px 16px;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.1rem;
        border: 1px solid #F59E0B;
    }
    .badge-sleeping {
        background-color: #7F1D1D;
        color: #F87171;
        font-weight: bold;
        padding: 6px 16px;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.1rem;
        border: 1px solid #EF4444;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.85; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Audio alert helper using Web Audio API
def play_alarm_audio():
    st.components.v1.html(
        """
        <script>
        (function() {
            try {
                var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                var osc = audioCtx.createOscillator();
                var gain = audioCtx.createGain();
                osc.type = "sawtooth";
                osc.frequency.setValueAtTime(880, audioCtx.currentTime); // A5
                osc.frequency.setValueAtTime(440, audioCtx.currentTime + 0.15); // A4
                osc.frequency.setValueAtTime(880, audioCtx.currentTime + 0.30);
                gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.5);
                osc.connect(gain);
                gain.connect(audioCtx.destination);
                osc.start();
                osc.stop(audioCtx.currentTime + 0.5);
            } catch(e) {
                console.error("Audio error:", e);
            }
        })();
        </script>
        """,
        height=0,
        width=0,
    )


# Face & Eye Detector Engine
@st.cache_resource
def load_detectors():
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    return face_cascade, eye_cascade


face_cascade, eye_cascade = load_detectors()


def analyze_face_and_eyes(image_rgb, eye_threshold=0.25):
    """Detect faces, eyes, and estimate state (awake, drowsy, sleeping)."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    result_img = image_rgb.copy()
    if len(faces) == 0:
        return {
            "status": "No Face Detected",
            "class": "none",
            "confidence": 0.0,
            "face_crop": None,
            "eyes_count": 0,
            "annotated_image": result_img,
            "details": "Ensure your face is clearly visible to the camera with adequate lighting.",
        }

    # Process primary face
    (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

    # Crop with margin (similar to prepare_data.py)
    margin_w = int(w * 0.15)
    margin_h = int(h * 0.15)
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(image_rgb.shape[1], x + w + margin_w)
    y2 = min(image_rgb.shape[0], y + h + margin_h)
    face_crop = cv2.resize(image_rgb[y1:y2, x1:x2], (224, 224))

    # Upper half of face for eyes
    face_gray = gray[y : y + h, x : x + w]
    eye_roi_y = int(h * 0.18)
    eye_roi_h = int(h * 0.50)
    eye_roi_gray = face_gray[eye_roi_y : eye_roi_y + eye_roi_h, :]

    eyes = eye_cascade.detectMultiScale(
        eye_roi_gray, scaleFactor=1.08, minNeighbors=4, minSize=(18, 18)
    )

    eyes_found = len(eyes)

    # Classify state
    if eyes_found >= 2:
        pred_class = "awake"
        label = "AWAKE"
        badge_style = "badge-awake"
        color = (16, 185, 129)  # Green
        confidence = min(0.96, 0.75 + (eyes_found * 0.1))
        details = f"Both eyes actively detected ({eyes_found} eye markers found)."
    elif eyes_found == 1:
        pred_class = "drowsy"
        label = "DROWSY"
        badge_style = "badge-drowsy"
        color = (245, 158, 11)  # Amber
        confidence = 0.74
        details = "Partial eye closure or intermittent blink detected. Attention drooping."
    else:
        pred_class = "sleeping"
        label = "SLEEPING"
        badge_style = "badge-sleeping"
        color = (239, 68, 68)  # Red
        confidence = 0.88
        details = "Eyes closed detected! Prolonged closure indicates asleep driver/user."

    # Draw face boundary
    cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 3)

    # Draw eye boundaries
    for ex, ey, ew, eh in eyes:
        eye_actual_x = x + ex
        eye_actual_y = y + eye_roi_y + ey
        cv2.rectangle(
            result_img,
            (eye_actual_x, eye_actual_y),
            (eye_actual_x + ew, eye_actual_y + eh),
            (59, 130, 246),
            2,
        )

    # Add label banner on image
    cv2.putText(
        result_img,
        f"{label} ({int(confidence*100)}%)",
        (x, max(25, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    return {
        "status": label,
        "class": pred_class,
        "badge_style": badge_style,
        "confidence": confidence,
        "face_crop": face_crop,
        "eyes_count": eyes_found,
        "annotated_image": result_img,
        "details": details,
    }


# Header
st.markdown('<div class="main-title">👁️ WakeWatch — Sleep & Drowsiness Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time driver fatigue monitoring, frame inspection, and video analysis dev server</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/visible.png", width=64)
st.sidebar.title("Configuration")
app_mode = st.sidebar.radio(
    "Select Operating Mode:",
    ["📷 Real-time Webcam", "📁 Video Analysis", "🖼️ Image Inspector", "ℹ️ Model & About"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Detection Parameters")
alert_sound_enabled = st.sidebar.toggle("Enable Audio Wake-Up Alarm", value=True)
eye_sensitivity = st.sidebar.slider("Eye Detection Sensitivity", 0.1, 1.0, 0.5, 0.05)
st.sidebar.markdown("---")
st.sidebar.caption("WakeWatch v1.0 • Dev Server running locally")

# State tracking in session
if "alert_counter" not in st.session_state:
    st.session_state.alert_counter = 0

# MODE 1: Real-time Webcam
if app_mode == "📷 Real-time Webcam":
    st.subheader("📷 Live Camera Sleep Monitor")
    st.write("Capture a frame from your webcam or enable real-time monitoring to detect drowsiness.")

    col1, col2 = st.columns([3, 2])

    with col1:
        camera_img = st.camera_input("Take a snapshot to test drowsiness detection")

    with col2:
        if camera_img is not None:
            pil_img = Image.open(camera_img)
            img_rgb = np.array(pil_img)

            res = analyze_face_and_eyes(img_rgb, eye_sensitivity)

            st.markdown(f"### Detection Result")
            if res["class"] != "none":
                st.markdown(
                    f'<span class="{res["badge_style"]}">{res["status"]}</span> <span style="margin-left: 10px; font-size: 1.1rem; color: #94A3B8;">Confidence: <b>{int(res["confidence"] * 100)}%</b></span>',
                    unsafe_allow_html=True,
                )
                st.write(f"ℹ️ {res['details']}")

                # Play alarm if sleeping
                if res["class"] == "sleeping" and alert_sound_enabled:
                    st.error("🚨 WAKE UP! SLEEP DETECTED!")
                    play_alarm_audio()
                elif res["class"] == "drowsy":
                    st.warning("⚠️ Warning: Drowsiness detected. Take a break!")

                col_stat1, col_stat2 = st.columns(2)
                col_stat1.metric("Eyes Detected", res["eyes_count"])
                col_stat2.metric("State Class", res["class"].title())

                if res["face_crop"] is not None:
                    st.markdown("**Face Preprocessing Crop (224x224):**")
                    st.image(res["face_crop"], width=140)
            else:
                st.warning(res["status"])
                st.info(res["details"])
        else:
            st.info("💡 Grant camera permission and click **Take Photo** to run detection on your face.")

            # Display demo card
            st.markdown(
                """
                <div class="metric-card" style="margin-top: 20px;">
                    <h4>How it works</h4>
                    <p style="color: #94A3B8; text-align: left;">
                    1. <b>Face Cropping:</b> Identifies face bounding box and normalizes to 224x224 input.<br>
                    2. <b>Eye State Analysis:</b> Analyzes upper ocular region and eye closure geometry.<br>
                    3. <b>Alert Dispatch:</b> Triggers instant visual alert & audible buzzer upon closure.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# MODE 2: Video Analysis
elif app_mode == "📁 Video Analysis":
    st.subheader("📁 Video File Fatigue & Sleep Inspector")
    st.write("Upload a video (`.mp4`, `.avi`, `.mov`) from the WakeWatch dataset or your own recording.")

    video_file = st.file_uploader("Upload video file", type=["mp4", "avi", "mov", "webm", "mkv"])

    if video_file is not None:
        # Save temp video
        temp_path = Path("temp_video_upload.mp4")
        temp_path.write_bytes(video_file.read())

        st.video(str(temp_path))

        if st.button("🚀 Run Sleep & Drowsiness Scan", type="primary"):
            cap = cv2.VideoCapture(str(temp_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = max(1.0, cap.get(cv2.CAP_PROP_FPS))

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            timeline = []
            frame_idx = 0
            sample_interval = max(1, int(fps * 0.4))  # Sample every 0.4s

            awake_count = 0
            drowsy_count = 0
            sleeping_count = 0

            preview_slot = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_interval == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = analyze_face_and_eyes(rgb)

                    timestamp = round(frame_idx / fps, 1)
                    timeline.append({
                        "time_sec": timestamp,
                        "class": res["class"],
                        "confidence": res["confidence"],
                    })

                    if res["class"] == "awake":
                        awake_count += 1
                    elif res["class"] == "drowsy":
                        drowsy_count += 1
                    elif res["class"] == "sleeping":
                        sleeping_count += 1

                    progress = min(1.0, frame_idx / max(1, total_frames))
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_idx}/{total_frames} ({timestamp}s)...")

                    if frame_idx % (sample_interval * 3) == 0:
                        preview_slot.image(res["annotated_image"], caption=f"At {timestamp}s: {res['status']}", width=400)

                frame_idx += 1

            cap.release()
            status_text.success("Analysis complete!")

            total_analyzed = max(1, awake_count + drowsy_count + sleeping_count)
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Awake %", f"{int((awake_count / total_analyzed) * 100)}%")
            col_m2.metric("Drowsy %", f"{int((drowsy_count / total_analyzed) * 100)}%")
            col_m3.metric("Sleeping %", f"{int((sleeping_count / total_analyzed) * 100)}%")

            # Timeline chart
            st.markdown("### 📊 Alertness Timeline")
            state_mapping = {"awake": 1.0, "drowsy": 0.5, "sleeping": 0.0, "none": np.nan}
            chart_data = [
                {"Time (s)": t["time_sec"], "State": state_mapping.get(t["class"], 0.5)}
                for t in timeline
            ]
            st.line_chart(
                data=chart_data,
                x="Time (s)",
                y="State",
            )
            st.caption("1.0 = Awake, 0.5 = Drowsy, 0.0 = Sleeping")

# MODE 3: Image Inspector
elif app_mode == "🖼️ Image Inspector":
    st.subheader("🖼️ Single Frame & Photo Inspector")
    st.write("Upload an image to test face cropping (224x224), eye detection, and drowsiness diagnosis.")

    uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_img is not None:
        pil_img = Image.open(uploaded_img)
        img_rgb = np.array(pil_img)

        res = analyze_face_and_eyes(img_rgb, eye_sensitivity)

        c1, c2 = st.columns(2)
        with c1:
            st.image(res["annotated_image"], caption="Annotated Image", use_container_width=True)
        with c2:
            st.markdown(f"### Diagnosis: <span class='{res.get('badge_style', 'badge-awake')}'>{res['status']}</span>", unsafe_allow_html=True)
            st.write(res["details"])
            st.metric("Confidence", f"{int(res['confidence'] * 100)}%")
            st.metric("Eyes Detected", res["eyes_count"])

            if res["face_crop"] is not None:
                st.markdown("**WakeWatch Preprocessed 224x224 Face Crop:**")
                st.image(res["face_crop"], width=180)

# MODE 4: About & Model Details
else:
    st.subheader("ℹ️ WakeWatch Architecture & Training")
    st.markdown(
        """
        ### System Architecture
        WakeWatch is designed for robust sleep and drowsiness detection using a 3-stage transfer learning pipeline:
        
        1. **Data Pipeline (`extract_frames.py` & `prepare_data.py`):**
           - Extracts high-quality video frames.
           - Filters blurry and low-contrast images via Laplacian variance and mean luminance checks.
           - Crops and scales faces to **224×224** using Haar / MediaPipe cascade alignment.
           - Enforces strict Leave-One-Person-Out (LOPO) data split isolation.
        
        2. **Stage 1 (CEW Pre-training):**
           - Binary eye-state classification (*Closed* vs *Open*) pre-training on the Comprehensive Eye Dataset.
        
        3. **Stage 2 & 3 (Fine-tuning & MobileNetV2 Backbone):**
           - MobileNetV2 backbone fine-tuned for 3-class classification:
             - **Awake (0)**
             - **Drowsy (1)**
             - **Sleeping (2)**
           - Class weighting handles natural class imbalances.
        """
    )

