import streamlit as st
import mediapipe as mp
import cv2
import av
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import time
import random
import base64
from io import BytesIO

from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, RTCConfiguration

# ── MediaPipe setup ────────────────────────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
MODEL_PATH = "pose_landmarker.task"

st.set_page_config(page_title="Just Graph", layout="centered")

# ── Session state defaults ─────────────────────────────────────────────────────
DEFAULTS = {
    "page": "home",
    "graphs_start_time": None,
    "cam_start_time": None,
    "frozen_frame": None,
    "frozen_landmarks": None,
    "current_graph": None,
    "score": None,
    "feedback": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Graph definitions ──────────────────────────────────────────────────────────
GRAPH_TYPES = {
    "linear": {
        "label": "Linear",
        "equation": "y = 2x + 1",
        "fn": lambda x: 2 * x + 1,
        "xlim": (-5, 5),
        "ylim": (-10, 10),
        # Expected arm shape description for scoring hints
        "description": "straight diagonal line going up-right",
    },
    "quadratic": {
        "label": "Quadratic",
        "equation": "y = x²",
        "fn": lambda x: x ** 2,
        "xlim": (-4, 4),
        "ylim": (-1, 12),
        "description": "U-shaped curve, arms raised on both sides",
    },
    "absolute_value": {
        "label": "Absolute Value",
        "equation": "y = |x|",
        "fn": lambda x: np.abs(x),
        "xlim": (-5, 5),
        "ylim": (-1, 6),
        "description": "V-shape, arms angled upward from center",
    },
    "square_root": {
        "label": "Square Root",
        "equation": "y = √x",
        "fn": lambda x: np.where(x >= 0, np.sqrt(np.maximum(x, 0)), np.nan),
        "xlim": (-1, 9),
        "ylim": (-1, 4),
        "description": "curve rising steeply then flattening, right side only",
    },
    "cubic": {
        "label": "Cubic",
        "equation": "y = x³",
        "fn": lambda x: x ** 3,
        "xlim": (-3, 3),
        "ylim": (-10, 10),
        "description": "S-shaped curve going from bottom-left to top-right",
    },
    "exponential": {
        "label": "Exponential",
        "equation": "y = 2ˣ",
        "fn": lambda x: np.power(2.0, x),
        "xlim": (-4, 4),
        "ylim": (-1, 12),
        "description": "flat on the left, rapidly rising on the right",
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def change_page(new_page):
    st.session_state.page = new_page

def clamp_score(value):
    return max(0, min(100, int(value)))

def make_graph_figure(graph_key: str, figsize=(6, 5)):
    """Return a matplotlib Figure for the given graph type."""
    g = GRAPH_TYPES[graph_key]
    x = np.linspace(g["xlim"][0], g["xlim"][1], 500)
    y = g["fn"](x)

    fig, ax = plt.subplots(figsize=figsize, facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    ax.plot(x, y, color="#00d4ff", linewidth=3, label=g["equation"])

    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#555")
    ax.spines["left"].set_color("#555")

    ax.tick_params(colors="#888", labelsize=9)
    ax.set_xlim(g["xlim"])
    ax.set_ylim(g["ylim"])
    ax.grid(True, color="#222", linewidth=0.5, linestyle="--")
    ax.set_xlabel("x", color="#888", fontsize=11)
    ax.set_ylabel("y", color="#888", fontsize=11, rotation=0, labelpad=12)
    ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="#00d4ff", fontsize=12)

    fig.tight_layout()
    return fig

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ── Scoring logic ──────────────────────────────────────────────────────────────
def score_pose(landmarks, graph_key):
    """
    Compare the user's arm pose to the expected shape of the graph.
    Returns (score: int, feedback: str).
    """
    lm = landmarks[0]

    # Key landmarks (normalized 0–1, y increases downward)
    nose          = lm[0]
    left_shoulder = lm[11]
    right_shoulder= lm[12]
    left_elbow    = lm[13]
    right_elbow   = lm[14]
    left_wrist    = lm[15]
    right_wrist   = lm[16]
    left_hip      = lm[23]
    right_hip     = lm[24]

    def arm_angle_deg(shoulder, elbow, wrist):
        """Angle of the full arm (shoulder→wrist vector) in degrees. 0=right, 90=up, -90=down."""
        dx = wrist.x - shoulder.x
        dy = shoulder.y - wrist.y  # flip y since screen y increases downward
        return np.degrees(np.arctan2(dy, dx))

    def elbow_bend(shoulder, elbow, wrist):
        """How much the elbow deviates from a straight shoulder→wrist line (0=straight)."""
        # Vector shoulder→wrist
        sw_dx = wrist.x - shoulder.x
        sw_dy = wrist.y - shoulder.y
        sw_len = np.sqrt(sw_dx**2 + sw_dy**2) + 1e-6
        # Project elbow onto that line
        t = ((elbow.x - shoulder.x) * sw_dx + (elbow.y - shoulder.y) * sw_dy) / (sw_len**2)
        proj_x = shoulder.x + t * sw_dx
        proj_y = shoulder.y + t * sw_dy
        # Perpendicular distance (normalized by body width)
        body_width = abs(left_shoulder.x - right_shoulder.x) + 1e-6
        return np.sqrt((elbow.x - proj_x)**2 + (elbow.y - proj_y)**2) / body_width

    # Arm angles
    left_angle  = arm_angle_deg(left_shoulder,  left_elbow,  left_wrist)
    right_angle = arm_angle_deg(right_shoulder, right_elbow, right_wrist)

    # Wrist height relative to shoulder (positive = wrist above shoulder)
    lw_rel = left_shoulder.y  - left_wrist.y
    rw_rel = right_shoulder.y - right_wrist.y
    le_rel = left_shoulder.y  - left_elbow.y
    re_rel = right_shoulder.y - right_elbow.y

    # Elbow bend (straightness)
    left_bend  = elbow_bend(left_shoulder,  left_elbow,  left_wrist)
    right_bend = elbow_bend(right_shoulder, right_elbow, right_wrist)
    arms_straight = left_bend < 0.25 and right_bend < 0.25

    # Horizontal wrist spread
    wrist_spread = abs(left_wrist.x - right_wrist.x)

    score = 0
    feedback_parts = []

    if graph_key == "linear":
        # Want one arm angled up (~30–80°) and other angled down (~-30 to -80°)
        # or both arms forming one continuous diagonal line
        angle_diff = abs(left_angle - right_angle)  # ideally ~120–180° apart
        high_arm   = max(lw_rel, rw_rel)
        low_arm    = min(lw_rel, rw_rel)
        diagonal   = high_arm > 0.12 and low_arm < -0.05

        if diagonal:
            score += 40
            feedback_parts.append("Good diagonal direction!")
        else:
            feedback_parts.append("One arm should be clearly up, the other clearly down.")

        # Reward how far apart the angles are (closer to 180° = straighter diagonal)
        angle_score = min(40, int((min(angle_diff, 160) / 160) * 40))
        score += angle_score

        # Penalise bent elbows
        if arms_straight:
            score += 20
            feedback_parts.append("Arms nice and straight!")
        else:
            score += 5
            feedback_parts.append("Try to straighten your arms more.")

    elif graph_key == "quadratic":
        both_up   = lw_rel > 0.08 and rw_rel > 0.08
        spread    = wrist_spread > 0.45

        if both_up:
            # How symmetrically are both arms raised?
            symmetry = 1 - min(abs(lw_rel - rw_rel) / 0.3, 1.0)
            score += int(40 * symmetry)
            feedback_parts.append("Both arms up — nice U-shape!" if symmetry > 0.6 else "Try to raise both arms equally.")
        else:
            feedback_parts.append("Raise both wrists above your shoulders for a U-shape.")

        if spread:
            score += 20
            feedback_parts.append("Good width!")
        else:
            feedback_parts.append("Spread your arms wider.")

        if arms_straight:
            score += 20
        else:
            score += 5
            feedback_parts.append("Straighter arms = cleaner parabola.")

        # Symmetry bonus
        if both_up and spread and arms_straight:
            score += 20

    elif graph_key == "absolute_value":
        both_up = lw_rel > 0.08 and rw_rel > 0.08
        spread  = wrist_spread > 0.5

        if both_up and spread:
            symmetry = 1 - min(abs(lw_rel - rw_rel) / 0.3, 1.0)
            score += int(50 * symmetry)
            feedback_parts.append("Excellent V-shape!" if symmetry > 0.6 else "Make sure both sides of the V are equal.")
        elif both_up:
            score += 25
            feedback_parts.append("Arms up but spread them wider for a sharper V.")
        else:
            feedback_parts.append("Raise both arms upward and outward.")

        if arms_straight:
            score += 25
            feedback_parts.append("Straight arms make a crisp V!")
        else:
            score += 5
            feedback_parts.append("Straighten your arms for a cleaner V.")

        if both_up and spread and arms_straight:
            score += 25

    elif graph_key == "square_root":
        # Right arm high, left arm low/neutral; right elbow should be fairly straight
        right_high  = rw_rel > 0.12
        left_low    = lw_rel < 0.06
        right_str   = right_bend < 0.25

        if right_high:
            height_score = min(35, int((min(rw_rel, 0.35) / 0.35) * 35))
            score += height_score
            feedback_parts.append("Right arm raised well!")
        else:
            feedback_parts.append("Raise your right arm higher.")

        if left_low:
            score += 25
            feedback_parts.append("Left side nicely low.")
        else:
            feedback_parts.append("Lower your left arm — the curve starts flat.")

        if right_str:
            score += 20
            feedback_parts.append("Right arm is straight — great!")
        else:
            score += 5

        if right_high and left_low:
            score += 20  # bonus for nailing the overall shape

    elif graph_key == "cubic":
        # S-curve: one wrist up, other down, AND elbows curve opposite direction
        s_wrists = (lw_rel > 0.08 and rw_rel < -0.05) or (rw_rel > 0.08 and lw_rel < -0.05)
        s_elbows = (le_rel < -0.02 and re_rel > 0.02) or (re_rel < -0.02 and le_rel > 0.02)

        if s_wrists:
            wrist_diff = abs(lw_rel - rw_rel)
            score += min(40, int((min(wrist_diff, 0.4) / 0.4) * 40))
            feedback_parts.append("Good S-curve wrist direction!")
        else:
            feedback_parts.append("One arm should go up, the other down for an S-shape.")

        if s_elbows:
            score += 30
            feedback_parts.append("Elbows curving correctly — nice S!")
        else:
            score += 5
            feedback_parts.append("Bend your elbows in opposite directions to show the S curve.")

        if s_wrists and s_elbows:
            score += 30  # full S bonus

    elif graph_key == "exponential":
        right_very_high = rw_rel > 0.18
        left_low        = lw_rel < 0.06
        right_str       = right_bend < 0.25

        if right_very_high:
            height_score = min(40, int((min(rw_rel, 0.4) / 0.4) * 40))
            score += height_score
            feedback_parts.append("Right arm high — great steep rise!")
        else:
            feedback_parts.append("Raise your right arm much higher for the steep part.")

        if left_low:
            score += 25
            feedback_parts.append("Left arm low — good flat start!")
        else:
            feedback_parts.append("Keep your left arm low and close to horizontal.")

        if right_str:
            score += 15

        if right_very_high and left_low:
            score += 20  # bonus for nailing the contrast

    score = clamp_score(score)
    feedback = " ".join(feedback_parts) if feedback_parts else "Keep practicing!"

    # Letter grade
    if score >= 90:
        grade = "A+ 🌟"
    elif score >= 80:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 60:
        grade = "C"
    else:
        grade = "D — keep trying!"

    return score, grade, feedback

# ── Video Processor ────────────────────────────────────────────────────────────
class PoseVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.pose = PoseLandmarker.create_from_options(self.options)
        self.start_time = time.monotonic()
        self.last_img = None
        self.last_landmarks = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.last_img = img.copy()

        timestamp_ms = int((time.monotonic() - self.start_time) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        results = self.pose.detect_for_video(mp_image, timestamp_ms)

        if results.pose_landmarks:
            self.last_landmarks = results.pose_landmarks
            h, w, _ = img.shape
            # Upper body landmarks only: shoulders(11,12), elbows(13,14), wrists(15,16), hips(23,24)
            UPPER_BODY = [11, 12, 13, 14, 15, 16, 23, 24]
            for idx in UPPER_BODY:
                lm = results.pose_landmarks[0][idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 6, (255, 20, 147), -1)
            # Draw arm + shoulder connections
            connections = [(11,13),(13,15),(12,14),(14,16),(11,12),(11,23),(12,24)]
            for a, b in connections:
                la = results.pose_landmarks[0][a]
                lb = results.pose_landmarks[0][b]
                cv2.line(img,
                    (int(la.x*w), int(la.y*h)),
                    (int(lb.x*w), int(lb.y*h)),
                    (0, 200, 255), 2)
        else:
            self.last_landmarks = None
            cv2.putText(img, "STEP BACK — can't see you!", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ── Pages ──────────────────────────────────────────────────────────────────────
def home():
    st.markdown("""
        <h1 style='text-align:center; font-size:3rem; color:#00d4ff;'>📈 Just Graph</h1>
        <p style='text-align:center; color:#aaa; font-size:1.1rem;'>
            A graph appears. Memorize its shape.<br>
            Then recreate it with your arms!
        </p>
    """, unsafe_allow_html=True)
    st.write("")
    st.write("Made by Sara Koka")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("▶ Start", use_container_width=True, type="primary"):
            st.session_state.current_graph = random.choice(list(GRAPH_TYPES.keys()))
            st.session_state.graphs_start_time = time.time()
            st.session_state.frozen_frame = None
            st.session_state.frozen_landmarks = None
            st.session_state.score = None
            change_page("graph")
            st.rerun()

def graph():
    GRAPH_DURATION = 5
    elapsed = time.time() - st.session_state.graphs_start_time
    remaining = max(0.0, GRAPH_DURATION - elapsed)

    g = GRAPH_TYPES[st.session_state.current_graph]

    st.markdown(f"<h2 style='text-align:center;'>Memorize this graph!</h2>", unsafe_allow_html=True)

    # Progress bar as timer
    st.progress(remaining / GRAPH_DURATION, text=f"⏱ {remaining:.1f}s remaining")

    fig = make_graph_figure(st.session_state.current_graph)
    st.pyplot(fig)
    plt.close(fig)

    st.info(f"**{g['label']}** — `{g['equation']}`")

    if remaining <= 0:
        st.session_state.cam_start_time = time.time()
        change_page("camera")
        st.rerun()
    else:
        time.sleep(0.25)
        st.rerun()

def camera():
    CAM_DURATION = 10
    g = GRAPH_TYPES[st.session_state.current_graph]

    st.markdown(f"<h2 style='text-align:center;'>Recreate with your arms!</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; color:#aaa;'>Graph: <b>{g['label']}</b> — <code>{g['equation']}</code></p>", unsafe_allow_html=True)

    # Show a small reminder of the graph
    with st.expander("👁 Remind me of the graph"):
        fig = make_graph_figure(st.session_state.current_graph, figsize=(4, 3))
        st.pyplot(fig)
        plt.close(fig)

    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    ctx = webrtc_streamer(
        key="pose",
        video_processor_factory=PoseVideoProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if st.session_state.cam_start_time is None:
        st.session_state.cam_start_time = time.time()

    elapsed = time.time() - st.session_state.cam_start_time
    remaining = max(0.0, CAM_DURATION - elapsed)

    st.progress(remaining / CAM_DURATION, text=f"⏱ {remaining:.1f}s to strike a pose!")

    if remaining <= 0:
        if ctx.video_processor and ctx.video_processor.last_img is not None:
            st.session_state.frozen_frame = ctx.video_processor.last_img.copy()
            st.session_state.frozen_landmarks = ctx.video_processor.last_landmarks
        change_page("accuracy")
        st.rerun()
    else:
        time.sleep(0.25)
        st.rerun()

def accuracy():
    g = GRAPH_TYPES[st.session_state.current_graph]

    st.markdown("<h2 style='text-align:center;'>Results 🎯</h2>", unsafe_allow_html=True)

    col_graph, col_pose = st.columns(2)

    with col_graph:
        st.markdown("**Target Graph**")
        fig = make_graph_figure(st.session_state.current_graph, figsize=(4,3))
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"{g['label']} — `{g['equation']}`")

    with col_pose:
        st.markdown("**Your Pose**")
        if st.session_state.frozen_frame is not None:
            st.image(st.session_state.frozen_frame, channels="BGR", use_container_width=True)
        else:
            st.warning("No frame captured.")

    st.divider()

    if st.session_state.frozen_landmarks:
        score, grade, feedback = score_pose(
            st.session_state.frozen_landmarks,
            st.session_state.current_graph
        )
        st.session_state.score = score

        # Score display
        color = "#00d4ff" if score >= 70 else "#ffd93d" if score >= 50 else "#ff6b6b"
        st.markdown(f"""
            <div style='text-align:center; padding:1.5rem; background:#1a1a2e;
                        border-radius:12px; border:1px solid {color};'>
                <div style='font-size:3rem; font-weight:bold; color:{color};'>{score}/100</div>
                <div style='font-size:2rem;'>{grade}</div>
                <div style='color:#ccc; margin-top:0.5rem;'>{feedback}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Couldn't detect your pose. Make sure your full upper body is visible!")

    st.write("")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🔄 Play Again", use_container_width=True, type="primary"):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            change_page("home")
            st.rerun()

# ── Router ─────────────────────────────────────────────────────────────────────
PAGE_MAP = {
    "home": home,
    "graph": graph,
    "camera": camera,
    "accuracy": accuracy,
}

PAGE_MAP[st.session_state.page]()