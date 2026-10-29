"""WakeWatch | Live Camera Sleep & Drowsiness Monitor.

Real-time driver fatigue monitoring with continuous WebRTC webcam streaming,
strict face-validation feedback, Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR),
time-based drowsiness state machine, and audio-visual sleep alarms.

Threading Model:
- Background WebRTC thread: Runs SleepVideoProcessor.recv(), detects faces,
  computes EAR/MAR, and updates SharedState under threading.Lock() on every frame.
- Main Streamlit thread: Polls SharedState snapshot while ctx.state.playing is True,
  dynamically re-rendering st.empty() telemetry placeholders every ~0.15s.
"""

import collections
import os
import threading
import time
import urllib.request
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

# ---------------------------------------------------------------------------
# Streamlit Page Setup & Custom Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="WakeWatch | Live Camera Sleep Monitor",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3B82F6, #10B981, #F59E0B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94A3B8;
        font-size: 1.0rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 0.85rem 1.0rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: border-color 0.2s ease-in-out;
    }
    .status-badge-awake {
        background-color: #065F46;
        color: #34D399;
        font-weight: bold;
        padding: 8px 18px;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.2rem;
        border: 1px solid #10B981;
    }
    .status-badge-drowsy {
        background-color: #78350F;
        color: #FBBF24;
        font-weight: bold;
        padding: 8px 18px;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.2rem;
        border: 1px solid #F59E0B;
    }
    .status-badge-sleeping {
        background-color: #7F1D1D;
        color: #F87171;
        font-weight: bold;
        padding: 8px 18px;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.2rem;
        border: 1px solid #EF4444;
        animation: pulse 1.2s infinite;
    }
    .status-badge-paused {
        background-color: #374151;
        color: #9CA3AF;
        font-weight: bold;
        padding: 8px 18px;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.2rem;
        border: 1px solid #6B7280;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.04); opacity: 0.85; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# STUN server configuration for seamless WebRTC connectivity
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
    }
)

# Ensure MediaPipe Face Landmarker model exists locally
MODEL_FILE = Path("face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not MODEL_FILE.exists():
    try:
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_FILE))
    except Exception as e:
        st.warning(f"Note: Could not pre-download model automatically: {e}")


# ---------------------------------------------------------------------------
# Audio Alarm Helper (Web Audio API Synthesizer)
# ---------------------------------------------------------------------------
def play_alarm_audio():
    """Generates a high-pitch pulsing siren using the client browser's Web Audio API."""
    st.components.v1.html(
        """
        <script>
        (function() {
            try {
                var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                if (audioCtx.state === 'suspended') {
                    audioCtx.resume();
                }
                var osc = audioCtx.createOscillator();
                var gain = audioCtx.createGain();
                osc.type = "sawtooth";
                osc.frequency.setValueAtTime(950, audioCtx.currentTime);
                osc.frequency.setValueAtTime(480, audioCtx.currentTime + 0.12);
                osc.frequency.setValueAtTime(950, audioCtx.currentTime + 0.25);
                gain.gain.setValueAtTime(0.4, audioCtx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.45);
                osc.connect(gain);
                gain.connect(audioCtx.destination);
                osc.start();
                osc.stop(audioCtx.currentTime + 0.45);
            } catch(e) {
                console.error("Audio buzzer error:", e);
            }
        })();
        </script>
        """,
        height=0,
        width=0,
    )


# ---------------------------------------------------------------------------
# Thread-Safe Shared State
# ---------------------------------------------------------------------------
class SharedState:
    """Thread-safe telemetry store shared between WebRTC thread and Streamlit UI.
    
    Protected by threading.Lock() to guarantee thread safety between the
    asynchronous video receiver thread and the Streamlit UI polling loop.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.status = "AWAKE"
        self.smoothed_ear = 0.0
        self.eyes_closed_sec = 0.0
        self.fps = 0.0
        self.face_valid = False
        self.face_hint = "Waiting for camera..."
        self.is_sleeping = False
        self.last_update = time.time()

    def update(self, status, ear, closed_sec, fps, valid, hint, sleeping):
        with self.lock:
            self.status = status
            self.smoothed_ear = ear
            self.eyes_closed_sec = closed_sec
            self.fps = fps
            self.face_valid = valid
            self.face_hint = hint
            self.is_sleeping = sleeping
            self.last_update = time.time()

    def reset(self):
        with self.lock:
            self.status = "AWAKE"
            self.smoothed_ear = 0.0
            self.eyes_closed_sec = 0.0
            self.fps = 0.0
            self.face_valid = False
            self.face_hint = "Waiting for camera..."
            self.is_sleeping = False
            self.last_update = time.time()

    def snapshot(self):
        with self.lock:
            return {
                "status": self.status,
                "smoothed_ear": self.smoothed_ear,
                "eyes_closed_sec": self.eyes_closed_sec,
                "fps": self.fps,
                "face_valid": self.face_valid,
                "face_hint": self.face_hint,
                "is_sleeping": self.is_sleeping,
            }


@st.cache_resource
def get_global_shared_state():
    """Provides a single persistent SharedState instance across Streamlit reruns."""
    return SharedState()


# Default metrics shown when the camera is idle or stopped
DEFAULT_SNAPSHOT = {
    "status": "AWAKE",
    "smoothed_ear": 0.0,
    "eyes_closed_sec": 0.0,
    "fps": 0.0,
    "face_valid": False,
    "face_hint": "Waiting for camera...",
    "is_sleeping": False,
}


# ---------------------------------------------------------------------------
# Module 1: Face Validator
# ---------------------------------------------------------------------------
class FaceValidator:
    """Validates face presence, bounding margins, size, and frontal orientation.
    
    Checks:
    a. Locks onto the primary face (largest area) and ignores background faces.
    b. Entire face bounding box is inside the frame with margin.
    c. Face size is within acceptable threshold (>= 15% width, <= 85% width).
    d. Face is reasonably frontal (yaw and pitch within tolerance) with both eyes visible.
    """

    def __init__(self, border_margin=18, min_width_ratio=0.15, max_width_ratio=0.85):
        self.border_margin = border_margin
        self.min_width_ratio = min_width_ratio
        self.max_width_ratio = max_width_ratio

    def validate(self, face_landmarks_list, frame_w, frame_h):
        if not face_landmarks_list or len(face_landmarks_list) == 0:
            return False, "No face detected", None, None

        # Lock onto the largest face by bounding box area
        primary_idx = 0
        max_area = -1
        best_bbox = None

        for idx, landmarks in enumerate(face_landmarks_list):
            xs = [int(lm.x * frame_w) for lm in landmarks]
            ys = [int(lm.y * frame_h) for lm in landmarks]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            area = (max_x - min_x) * (max_y - min_y)
            if area > max_area:
                max_area = area
                primary_idx = idx
                best_bbox = (min_x, min_y, max_x, max_y)

        landmarks = face_landmarks_list[primary_idx]
        min_x, min_y, max_x, max_y = best_bbox
        face_w = max_x - min_x

        # Criterion 1: Margin & Border Cut-Off
        if (
            min_x < self.border_margin
            or min_y < self.border_margin
            or max_x > (frame_w - self.border_margin)
            or max_y > (frame_h - self.border_margin)
        ):
            return False, "Face cut off - center face in frame", best_bbox, landmarks

        # Criterion 2: Face Size Check
        face_ratio = face_w / float(frame_w)
        if face_ratio < self.min_width_ratio:
            return False, "Move closer to camera", best_bbox, landmarks
        if face_ratio > self.max_width_ratio:
            return False, "Move back from camera", best_bbox, landmarks

        # Criterion 3: Head Pose & Frontal Orientation (Yaw & Pitch)
        nose = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        chin = landmarks[152]

        d_left = abs(nose.x - left_eye.x)
        d_right = abs(right_eye.x - nose.x)

        if d_right > 1e-4 and d_left > 1e-4:
            yaw_ratio = d_left / d_right
            if yaw_ratio < 0.38 or yaw_ratio > 2.65:
                return False, "Face the camera (head turned)", best_bbox, landmarks

        eye_mid_y = (left_eye.y + right_eye.y) / 2.0
        d_eye_nose = abs(nose.y - eye_mid_y)
        d_nose_chin = abs(chin.y - nose.y)

        if d_nose_chin > 1e-4:
            pitch_ratio = d_eye_nose / d_nose_chin
            if pitch_ratio < 0.20:
                return False, "Face the camera (tilted up)", best_bbox, landmarks
            if pitch_ratio > 3.0:
                return False, "Face the camera (tilted down)", best_bbox, landmarks

        return True, "Face Captured", best_bbox, landmarks


# ---------------------------------------------------------------------------
# Module 2: Ocular & Facial Geometry Analyzer
# ---------------------------------------------------------------------------
class OcularAnalyzer:
    """Calculates Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and maintains moving averages."""

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self, window_size=6):
        self.ear_history = collections.deque(maxlen=window_size)

    def _calc_single_ear(self, landmarks, indices, w, h):
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h_dist = np.linalg.norm(pts[0] - pts[3])
        if h_dist < 1e-5:
            return 0.0
        return float((v1 + v2) / (2.0 * h_dist))

    def compute_ear(self, landmarks, w, h):
        left_ear = self._calc_single_ear(landmarks, self.LEFT_EYE, w, h)
        right_ear = self._calc_single_ear(landmarks, self.RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        self.ear_history.append(avg_ear)
        smoothed_ear = float(np.mean(self.ear_history))
        return avg_ear, smoothed_ear

    def compute_mar(self, landmarks, w, h):
        top = np.array([landmarks[13].x * w, landmarks[13].y * h])
        bot = np.array([landmarks[14].x * w, landmarks[14].y * h])
        left = np.array([landmarks[61].x * w, landmarks[61].y * h])
        right = np.array([landmarks[291].x * w, landmarks[291].y * h])
        v_dist = np.linalg.norm(top - bot)
        h_dist = np.linalg.norm(left - right)
        if h_dist < 1e-5:
            return 0.0
        return float(v_dist / h_dist)

    def reset(self):
        self.ear_history.clear()


# ---------------------------------------------------------------------------
# Module 3: Drowsiness State Machine
# ---------------------------------------------------------------------------
class DrowsinessStateMachine:
    """Time-based fatigue & sleep state machine (independent of frame rate)."""

    def __init__(self):
        self.state = "AWAKE"
        self.closed_start_time = None
        self.eyes_closed_duration = 0.0
        self.yawn_start_time = None
        self.is_yawning = False

    def update(
        self,
        is_face_valid,
        smoothed_ear,
        mar,
        ear_threshold=0.23,
        drowsy_duration=1.2,
        sleeping_duration=3.0,
        enable_yawn=True,
    ):
        now = time.time()

        # If face is NOT properly captured, pause detection and reset timers
        if not is_face_valid:
            self.closed_start_time = None
            self.eyes_closed_duration = 0.0
            self.yawn_start_time = None
            self.is_yawning = False
            self.state = "PAUSED"
            return self.state, self.eyes_closed_duration

        # Yawn detection via MAR (> 0.65 for over 1.0 second)
        if enable_yawn and mar > 0.65:
            if self.yawn_start_time is None:
                self.yawn_start_time = now
            elif (now - self.yawn_start_time) > 1.0:
                self.is_yawning = True
        else:
            self.yawn_start_time = None
            self.is_yawning = False

        # Eye Closure analysis
        if smoothed_ear < ear_threshold:
            if self.closed_start_time is None:
                self.closed_start_time = now
            self.eyes_closed_duration = now - self.closed_start_time

            if self.eyes_closed_duration >= sleeping_duration:
                self.state = "SLEEPING"
            elif self.eyes_closed_duration >= drowsy_duration:
                self.state = "DROWSY"
            else:
                self.state = "AWAKE"
        else:
            self.closed_start_time = None
            self.eyes_closed_duration = 0.0

            if self.is_yawning:
                self.state = "DROWSY"
            else:
                self.state = "AWAKE"

        return self.state, self.eyes_closed_duration


# ---------------------------------------------------------------------------
# Module 4: Overlay & HUD Renderer
# ---------------------------------------------------------------------------
class OverlayRenderer:
    """Renders face bounding boxes, status badges, telemetry HUD, and alert banners."""

    COLOR_GREEN = (0, 230, 118)  # Captured / Awake
    COLOR_AMBER = (0, 180, 255)  # Drowsy / Adjust
    COLOR_RED = (40, 40, 240)    # Sleeping / Invalid
    COLOR_GRAY = (120, 120, 120) # Paused

    def draw(
        self,
        frame,
        is_face_valid,
        face_hint,
        bbox,
        landmarks,
        status,
        smoothed_ear,
        eyes_closed_sec,
        fps,
    ):
        h, w = frame.shape[:2]

        # 1. Face Bounding Box & Status Indicator
        if bbox is not None:
            min_x, min_y, max_x, max_y = bbox
            box_color = self.COLOR_GREEN if is_face_valid else self.COLOR_RED

            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), box_color, 2)
            corner_len = int(min(max_x - min_x, max_y - min_y) * 0.2)
            # Accent corners
            cv2.line(frame, (min_x, min_y), (min_x + corner_len, min_y), box_color, 4)
            cv2.line(frame, (min_x, min_y), (min_x, min_y + corner_len), box_color, 4)
            cv2.line(frame, (max_x, min_y), (max_x - corner_len, min_y), box_color, 4)
            cv2.line(frame, (max_x, min_y), (max_x, min_y + corner_len), box_color, 4)
            cv2.line(frame, (min_x, max_y), (min_x + corner_len, max_y), box_color, 4)
            cv2.line(frame, (min_x, max_y), (min_x, max_y - corner_len), box_color, 4)
            cv2.line(frame, (max_x, max_y), (max_x - corner_len, max_y), box_color, 4)
            cv2.line(frame, (max_x, max_y), (max_x, max_y - corner_len), box_color, 4)

            # Box Tag Header
            tag_text = "FACE CAPTURED" if is_face_valid else f"ADJUST POSITION: {face_hint.upper()}"
            tag_size, _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            tag_w, tag_h = tag_size[0] + 16, tag_size[1] + 10
            tag_y1 = max(0, min_y - tag_h)
            cv2.rectangle(frame, (min_x, tag_y1), (min_x + tag_w, min_y), box_color, -1)
            cv2.putText(
                frame,
                tag_text,
                (min_x + 8, min_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 0, 0) if is_face_valid else (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # 2. Eye Landmark Overlay when valid
        if landmarks is not None and is_face_valid:
            for idx in OcularAnalyzer.LEFT_EYE + OcularAnalyzer.RIGHT_EYE:
                px = int(landmarks[idx].x * w)
                py = int(landmarks[idx].y * h)
                cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)

        # 3. Top HUD Bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 54), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Status badge color
        if status == "AWAKE":
            badge_color = self.COLOR_GREEN
        elif status == "DROWSY":
            badge_color = self.COLOR_AMBER
        elif status == "SLEEPING":
            badge_color = self.COLOR_RED
        else:
            badge_color = self.COLOR_GRAY

        cv2.putText(
            frame,
            f"STATUS: {status}",
            (16, 36),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            badge_color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"EAR: {smoothed_ear:.3f}",
            (int(w * 0.38), 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"CLOSED: {eyes_closed_sec:.1f}s",
            (int(w * 0.60), 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (220, 220, 220) if eyes_closed_sec < 1.0 else self.COLOR_AMBER,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.0f}",
            (w - 110, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

        # 4. On-Screen Messages & Alerts
        if not is_face_valid:
            cv2.putText(
                frame,
                "Detection paused: face not properly captured",
                (20, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                self.COLOR_AMBER,
                2,
                cv2.LINE_AA,
            )
            if bbox is None:
                cv2.rectangle(frame, (int(w * 0.25), int(h * 0.40)), (int(w * 0.75), int(h * 0.58)), (20, 20, 180), -1)
                cv2.putText(
                    frame,
                    "NO FACE DETECTED",
                    (int(w * 0.32), int(h * 0.49)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.85,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "Please look directly at the camera",
                    (int(w * 0.31), int(h * 0.54)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (200, 200, 255),
                    1,
                    cv2.LINE_AA,
                )

        elif status == "SLEEPING":
            if int(time.time() * 3.5) % 2 == 0:
                alert_overlay = frame.copy()
                cv2.rectangle(alert_overlay, (0, 54), (w, 140), (20, 20, 220), -1)
                cv2.addWeighted(alert_overlay, 0.85, frame, 0.15, 0, frame)
                cv2.putText(
                    frame,
                    "! SLEEP ALERT ! WAKE UP !",
                    (int(w * 0.18), 112),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.1,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )

        elif status == "DROWSY":
            cv2.putText(
                frame,
                "WARNING: DROWSINESS DETECTED",
                (int(w * 0.22), 95),
                cv2.FONT_HERSHEY_DUPLEX,
                0.78,
                self.COLOR_AMBER,
                2,
                cv2.LINE_AA,
            )

        return frame


# ---------------------------------------------------------------------------
# Module 5: VideoProcessor (WebRTC Worker Thread)
# ---------------------------------------------------------------------------
class SleepVideoProcessor(VideoProcessorBase):
    """Continuous WebRTC frame processor for real-time drowsiness detection.
    
    IMPORTANT: This class executes inside the WebRTC background worker thread.
    It updates SharedState on EVERY frame under a lock and NEVER calls any st.*
    APIs or touches st.session_state directly.
    """

    def __init__(self, shared_state: SharedState):
        self.shared_state = shared_state
        self.validator = FaceValidator()
        self.ocular = OcularAnalyzer()
        self.state_machine = DrowsinessStateMachine()
        self.renderer = OverlayRenderer()

        # Dynamic parameter controls (live updated from Streamlit sliders)
        self.ear_threshold = 0.23
        self.drowsy_duration = 1.2
        self.sleeping_duration = 3.0
        self.enable_yawn = True

        # FPS calculation using rolling average over timestamps
        self.prev_time = None
        self.fps_history = collections.deque(maxlen=15)
        self.fps = 0.0

        # Lazy initialize MediaPipe Face Landmarker
        self._detector = None
        self._init_detector()

    def _init_detector(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision

            model_path = str(MODEL_FILE.resolve())
            if not os.path.exists(model_path):
                return

            base_options = BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                num_faces=3,
                min_face_detection_confidence=0.45,
                min_face_presence_confidence=0.45,
            )
            self._detector = vision.FaceLandmarker.create_from_options(options)
        except Exception:
            self._detector = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        h, w = img_bgr.shape[:2]

        # 1. Precise rolling FPS calculation
        now = time.time()
        if self.prev_time is not None:
            dt = now - self.prev_time
            if dt > 0:
                self.fps_history.append(1.0 / dt)
                self.fps = float(np.mean(self.fps_history))
        self.prev_time = now

        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        landmarks_list = []
        if self._detector is not None:
            try:
                import mediapipe as mp

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                detection_result = self._detector.detect(mp_image)
                if detection_result.face_landmarks:
                    landmarks_list = detection_result.face_landmarks
            except Exception:
                pass

        # 2. Face validation
        is_valid, hint, bbox, primary_landmarks = self.validator.validate(landmarks_list, w, h)

        # 3. Ocular & Fatigue analysis
        smoothed_ear = 0.0
        mar = 0.0
        if is_valid and primary_landmarks is not None:
            _, smoothed_ear = self.ocular.compute_ear(primary_landmarks, w, h)
            mar = self.ocular.compute_mar(primary_landmarks, w, h)
        else:
            self.ocular.reset()

        # 4. State machine update
        status, closed_sec = self.state_machine.update(
            is_valid,
            smoothed_ear,
            mar,
            ear_threshold=self.ear_threshold,
            drowsy_duration=self.drowsy_duration,
            sleeping_duration=self.sleeping_duration,
            enable_yawn=self.enable_yawn,
        )

        is_sleeping = (status == "SLEEPING")

        # 5. Render visual annotations on video
        annotated = self.renderer.draw(
            img_bgr,
            is_valid,
            hint,
            bbox,
            primary_landmarks,
            status,
            smoothed_ear,
            closed_sec,
            self.fps,
        )

        # 6. Push telemetry snapshot to SharedState on EVERY frame
        self.shared_state.update(
            status=status,
            ear=smoothed_ear,
            closed_sec=closed_sec,
            fps=self.fps,
            valid=is_valid,
            hint=hint,
            sleeping=is_sleeping,
        )

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ---------------------------------------------------------------------------
# Telemetry Renderer Helper
# ---------------------------------------------------------------------------
def render_telemetry_panel(container, snap, ear_threshold, sleep_duration, sound_enabled):
    """Renders the synchronized telemetry panel widgets into an st.empty container."""
    with container.container():
        st_status = snap["status"]
        if st_status == "AWAKE":
            badge_html = f'<span class="status-badge-awake">🟢 {st_status}</span>'
        elif st_status == "DROWSY":
            badge_html = f'<span class="status-badge-drowsy">🟡 {st_status} (WARNING)</span>'
        elif st_status == "SLEEPING":
            badge_html = f'<span class="status-badge-sleeping">🔴 {st_status} (ALARM)</span>'
        else:
            badge_html = f'<span class="status-badge-paused">⚪ {st_status} (NOT CAPTURED)</span>'

        st.markdown(badge_html, unsafe_allow_html=True)
        st.write("")

        # 2x2 Telemetry metrics grid
        m1, m2 = st.columns(2)
        m1.metric("Smoothed EAR", f"{snap['smoothed_ear']:.3f}", delta=f"Thresh: {ear_threshold:.2f}")
        m2.metric("Eyes Closed", f"{snap['eyes_closed_sec']:.1f} s", delta=f"Alarm: {sleep_duration:.1f}s")

        m3, m4 = st.columns(2)
        m3.metric("Video FPS", f"{snap['fps']:.0f} fps")
        face_valid_label = "✅ Valid" if snap["face_valid"] else "❌ Invalid"
        m4.metric("Face Validation", face_valid_label)

        # Face diagnostic card with dynamic border highlight
        border_col = "rgba(16, 185, 129, 0.5)" if snap["face_valid"] else "rgba(239, 68, 68, 0.5)"
        text_col = "#34D399" if snap["face_valid"] else "#F87171"
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top: 14px; border: 1.5px solid {border_col};">
                <b>Face Capture Diagnostic:</b><br>
                <span style="color: {text_col}; font-weight: 600; font-size: 1.05rem;">
                    {snap['face_hint']}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Audio alert trigger
        if snap["is_sleeping"] and sound_enabled:
            st.error("🚨 SLEEP ALERT: Driver/User Asleep! Wake Up!")
            play_alarm_audio()
        elif snap["status"] == "DROWSY":
            st.warning("⚠️ Warning: Drowsiness or prolonged blink detected.")


# ---------------------------------------------------------------------------
# Streamlit Main UI
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">👁️ WakeWatch — Live Camera Sleep Monitor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Continuous real-time webcam fatigue detection with strict face-capture validation</div>', unsafe_allow_html=True)

# Shared state singleton
shared_state = get_global_shared_state()

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/fluency/96/visible.png", width=64)
st.sidebar.title("Monitor Controls")

ear_thresh = st.sidebar.slider(
    "Eye Aspect Ratio (EAR) Threshold",
    min_value=0.15,
    max_value=0.35,
    value=0.23,
    step=0.01,
    help="EAR values below this threshold count as closed eyes (typical awake EAR is 0.28-0.34).",
)

drowsy_dur = st.sidebar.slider(
    "Drowsy Duration Threshold (seconds)",
    min_value=0.5,
    max_value=3.0,
    value=1.2,
    step=0.1,
    help="Continuous eye closure duration that triggers a DROWSY warning.",
)

sleep_dur = st.sidebar.slider(
    "Sleeping Duration Threshold (seconds)",
    min_value=1.5,
    max_value=6.0,
    value=3.0,
    step=0.2,
    help="Continuous eye closure duration that triggers the full SLEEP alarm.",
)

enable_sound = st.sidebar.toggle("Enable Audio Wake-Up Alarm", value=True)
enable_yawn_det = st.sidebar.toggle("Enable Yawn Detection (MAR)", value=True)

with st.sidebar.expander("🛠️ How to Tune Thresholds"):
    st.markdown(
        """
        - **EAR Threshold (0.20 - 0.26):**
          If the system reports you as sleeping while your eyes are open, lower the EAR threshold (e.g., 0.21). If it misses closed eyes, raise it (e.g., 0.25).
        - **Drowsy Duration (1.0s - 1.5s):**
          Natural blinks last 0.15 - 0.35s. Setting 1.2s avoids false alarms from deliberate blinks.
        - **Sleeping Duration (2.5s - 4.0s):**
          3.0 seconds is standard for driver micro-sleep prevention.
        - **Face Capture Status:**
          Detection only runs when the indicator is **GREEN** ("Face Captured"). If **RED**, follow the on-screen hint.
        """
    )

st.sidebar.markdown("---")
st.sidebar.caption("WakeWatch v2.1 • WebRTC Real-Time Engine")

# Main Layout: Live Stream & Real-Time Telemetry
col_video, col_telemetry = st.columns([7, 5])

with col_video:
    st.markdown("### 📹 Live Webcam Stream")
    ctx = webrtc_streamer(
        key="live-sleep-monitor",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=lambda: SleepVideoProcessor(shared_state=shared_state),
        media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
        async_processing=True,
    )

with col_telemetry:
    st.markdown("### 📊 Live Telemetry & Status")
    telemetry_placeholder = st.empty()

    # =======================================================================
    # THREADING MODEL & TELEMETRY SYNCHRONIZATION:
    # -----------------------------------------------------------------------
    # 1. Background WebRTC Worker:
    #    `SleepVideoProcessor.recv()` runs asynchronously in a worker thread.
    #    On every frame, it computes EAR, MAR, rolling FPS, and face validation,
    #    updating `SharedState` under a lock.
    #
    # 2. Main Streamlit UI Polling Loop:
    #    While `ctx.state.playing` is True, this while loop polls `SharedState`
    #    and re-renders `telemetry_placeholder` every ~0.15 seconds.
    #    It also passes live slider thresholds directly to the running processor.
    #
    # 3. Clean Stream Teardown:
    #    When the stream stops or before starting, `DEFAULT_SNAPSHOT` is rendered,
    #    resetting all metrics cleanly to defaults.
    # =======================================================================
    if ctx.state.playing:
        while ctx.state.playing:
            if ctx.video_processor:
                # Dynamically forward sidebar slider values to the active processor
                ctx.video_processor.ear_threshold = ear_thresh
                ctx.video_processor.drowsy_duration = drowsy_dur
                ctx.video_processor.sleeping_duration = sleep_dur
                ctx.video_processor.enable_yawn = enable_yawn_det

                snap = ctx.video_processor.shared_state.snapshot()
                render_telemetry_panel(telemetry_placeholder, snap, ear_thresh, sleep_dur, enable_sound)
            time.sleep(0.15)

        # Stream stopped: reset panel to default snapshot
        shared_state.reset()
        render_telemetry_panel(telemetry_placeholder, DEFAULT_SNAPSHOT, ear_thresh, sleep_dur, enable_sound)
    else:
        # Idle / Before stream starts
        render_telemetry_panel(telemetry_placeholder, DEFAULT_SNAPSHOT, ear_thresh, sleep_dur, enable_sound)

# ---------------------------------------------------------------------------
# Additional Tools Expander (Preserving Video & Image Inspection)
# ---------------------------------------------------------------------------
with st.expander("📁 Video & Single-Image Tools (Dataset Validation)"):
    st.markdown("Use these offline tools for dataset verification or batch video checks.")
    tab_vid, tab_img = st.tabs(["Video File Analysis", "Single Image Test"])

    with tab_vid:
        uploaded_vid = st.file_uploader("Upload video file (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
        if uploaded_vid is not None:
            temp_vid = Path("temp_uploaded_video.mp4")
            temp_vid.write_bytes(uploaded_vid.read())
            st.video(str(temp_vid))

    with tab_img:
        uploaded_img = st.file_uploader("Upload single frame / photo", type=["jpg", "png", "jpeg"])
        if uploaded_img is not None:
            pil_img = Image.open(uploaded_img)
            st.image(pil_img, caption="Uploaded Photo", width=360)
