"""
FormFlex — AI Squat Analyzer
==============================
Full-stack implementation matching the project abstract:

  Input     → Streamlit webcam/video @ 30 FPS target
  Detection → MediaPipe 33-point skeletal tracking
  Analysis  → 11 joint angles (atan2) fed into Random Forest .pkl
  Output    → Visual overlays (skeleton, angles, depth bar, feedback)
               + Pygame audio alerts (multi-modal)
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import pickle

from pose_estimation_module import PoseDetector
import audio_feedback as audio

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FormFlex — AI Squat Analyzer",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# CUSTOM CSS — Premium Dark Theme
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Base */
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: #0d0d0d; color: #f0f0f0; }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #141414; border-right: 1px solid #222; }

  /* KPI cards */
  .kpi-card {
    background: linear-gradient(135deg,#1a1a2e,#16213e);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 12px;
    text-align: center;
  }
  .kpi-label { font-size: 0.75rem; color: #888; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 4px; }
  .kpi-value { font-size: 2.4rem; font-weight: 700; color: #00d4ff; }
  .kpi-value.bad  { color: #ff4444; }
  .kpi-value.warn { color: #ff9900; }
  .kpi-value.good { color: #00ff88; }

  /* Feedback banner */
  .feedback-box {
    border-radius: 10px;
    padding: 14px 20px;
    font-size: 1.0rem;
    font-weight: 600;
    margin-top: 8px;
    border: 1px solid transparent;
  }
  .feedback-good   { background:#0d3323; border-color:#00ff88; color:#00ff88; }
  .feedback-warn   { background:#3d2800; border-color:#ff9900; color:#ff9900; }
  .feedback-bad    { background:#3d0000; border-color:#ff4444; color:#ff4444; }
  .feedback-neutral{ background:#1a1a1a; border-color:#555; color:#ccc; }

  /* Correction tip */
  .correction-tip {
    background: #1a1a2e;
    border-left: 4px solid #0f3460;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.87rem;
    color: #aac4ff;
    margin-top: 6px;
  }

  /* Angle table */
  .angle-row { display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #222; }
  .angle-name { color: #aaa; font-size:0.82rem; }
  .angle-val  { font-size:0.82rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# LOAD ML MODEL
# ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open('squat.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.sidebar.error(f"ML model not found: {e}")
        return None

ml_model = load_model()

# ──────────────────────────────────────────────────────────────────
# SQUAT-SPECIFIC FEEDBACK RULES
# Maps ML class label → (short status, detailed correction tip)
# ──────────────────────────────────────────────────────────────────
FEEDBACK_RULES = {
    "s_correct": {
        "status": "✅  Perfect Form",
        "tip":    "Great job! Maintain this position — knees tracking over toes, chest up, weight in heels.",
        "level":  "good",
    },
    "s_spine_neutral": {
        "status": "✅  Spine Neutral",
        "tip":    "Excellent back position! Keep your core braced and gaze slightly forward throughout the movement.",
        "level":  "good",
    },
    "s_caved_in_knees": {
        "status": "⚠️  Knees Caving In (Valgus Collapse)",
        "tip":    "Push your knees outward in line with your toes. Strengthen hip abductors (glutes). Try wider stance or turn toes out slightly.",
        "level":  "bad",
    },
    "s_feet_spread": {
        "status": "⚠️  Stance Too Wide / Feet Spread",
        "tip":    "Bring your feet closer — roughly shoulder-width apart, toes turned 15–30° outward. A too-wide stance reduces quad engagement.",
        "level":  "warn",
    },
}

MATH_RULES = {
    "lean_forward": {
        "status": "🚨  Leaning Too Far Forward",
        "tip":    "Keep your torso more upright. Engage your core, push hips back first, and keep weight in your mid-foot/heel. Consider mobility work in ankles/hips.",
        "level":  "bad",
    },
    "too_deep": {
        "status": "⚠️  Excessive Depth (Butt Wink Risk)",
        "tip":    "Squat only to parallel or slightly below. Going too deep causes posterior pelvic tilt (butt wink), stressing the lumbar spine.",
        "level":  "warn",
    },
    "going_down": {
        "status": "🔽  Descending",
        "tip":    "Good — control the descent. Take 2–3 seconds going down. Keep knees tracking toes.",
        "level":  "neutral",
    },
    "coming_up": {
        "status": "🔼  Ascending",
        "tip":    "Drive through your heels. Squeeze glutes at the top. Avoid letting the chest drop.",
        "level":  "neutral",
    },
    "standing": {
        "status": "🧍  Standing — Ready",
        "tip":    "Brace your core, set your stance, and begin the squat when ready.",
        "level":  "neutral",
    },
    "perfect_depth": {
        "status": "⬇️  Perfect Depth Reached",
        "tip":    "Great depth! Pause briefly, then drive up powerfully.",
        "level":  "good",
    },
}

# ──────────────────────────────────────────────────────────────────
# DETECTOR (cached so it's not re-created every Streamlit rerun)
# ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_detector():
    return PoseDetector(detectionCon=0.65, trackCon=0.65)

detector = get_detector()

# ──────────────────────────────────────────────────────────────────
# 11 JOINT ANGLES SPEC
# Each tuple: (name, p1, vertex, p3, draw?)
# ──────────────────────────────────────────────────────────────────
JOINT_ANGLES = [
    # Lower Body — primary squat joints
    ("R Knee",    24, 26, 28, True),    # R Hip - R Knee - R Ankle
    ("L Knee",    23, 25, 27, True),    # L Hip - L Knee - L Ankle
    ("R Hip",     12, 24, 26, True),    # R Shoulder - R Hip - R Knee
    ("L Hip",     11, 23, 25, True),    # L Shoulder - L Hip - L Knee
    ("R Ankle",   26, 28, 32, True),    # R Knee - R Ankle - R Foot
    ("L Ankle",   25, 27, 31, True),    # L Knee - L Ankle - L Foot
    # Upper Body — posture/torso checks
    ("R Shoulder",14, 12, 24, False),   # R Elbow - R Shoulder - R Hip
    ("L Shoulder",13, 11, 23, False),   # L Elbow - L Shoulder - L Hip
    ("R Elbow",   12, 14, 16, False),   # R Shoulder - R Elbow - R Wrist
    ("L Elbow",   11, 13, 15, False),   # L Shoulder - L Elbow - L Wrist
    ("Spine Lean", 0, 12, 24, True),    # Nose - R Shoulder - R Hip (torso tilt)
]

# ──────────────────────────────────────────────────────────────────
# CORE FRAME PROCESSING
# ──────────────────────────────────────────────────────────────────
def process_frame(img: np.ndarray, state: dict) -> tuple:
    """
    Runs full analysis pipeline on one frame.
    Returns (annotated_img, angles_dict, rule_key)
    """
    img = cv2.resize(img, (720, 540))

    # 1 — Pose detection
    detector.findPose(img)
    lmList = detector.findPosition(img)

    # Default outputs
    angles = {}
    rule_key = "standing"

    if len(lmList) < 29:   # need at minimum hip→ankle landmarks
        # Draw a "no person detected" overlay
        cv2.putText(img, "No person detected — Stand in frame", (30, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        return img, angles, rule_key

    # 2 — Draw full 33-point skeleton
    detector.drawSkeleton(img, line_thickness=2)

    # 3 — Calculate all 11 joint angles
    for (name, p1, p2, p3, draw) in JOINT_ANGLES:
        a = detector.findAngle(img, p1, p2, p3, draw=draw, label="")
        angles[name] = round(a, 1)

    r_knee = angles.get("R Knee", 0)

    # 4 — Depth percentage (160° = standing, 70° = parallel)
    depth_pct = float(np.interp(r_knee, (70, 160), (100, 0)))
    depth_pct = max(0.0, min(100.0, depth_pct))
    bar_y = int(np.interp(r_knee, (70, 160), (480, 200)))

    # 5 — Rep counting
    # Thresholds: 80% = squat bottom reached, 20% = returned to standing
    # These are deliberately loose so normal squatting depth always triggers counting
    if depth_pct >= 80 and state["dir"] == 0:
        state["dir"] = 1        # mark as descended
    if depth_pct <= 20 and state["dir"] == 1:
        state["count"] += 1     # count full rep when they stand back up
        state["dir"] = 0
        audio.play("rep")

    # 6 — ML classification
    ml_rule = None
    raw_feat = detector.get_raw_landmarks_features()
    if ml_model and len(raw_feat) == 132:
        pred = ml_model.predict([raw_feat])[0]
        if pred in FEEDBACK_RULES:
            ml_rule = pred

    # 7 — Determine feedback rule (ML first, then math fallback)
    hip_angle = angles.get("R Hip", 90)

    if ml_rule and ml_rule not in ("s_correct", "s_spine_neutral"):
        rule_key = ml_rule
        audio.play("warning")
    elif hip_angle < 45:
        rule_key = "lean_forward"
        audio.play("warning")
    elif r_knee < 50:
        rule_key = "too_deep"
        audio.play("warning")
    elif depth_pct >= 80:
        rule_key = "perfect_depth"
        audio.play("good")
    elif depth_pct <= 20:
        rule_key = "standing"
    elif state["dir"] == 0:
        rule_key = "going_down"   # heading toward squat bottom
    else:
        rule_key = "coming_up"    # heading back up

    # 8 — Draw HUD overlays on frame
    _draw_hud(img, state, depth_pct, bar_y, rule_key, r_knee, hip_angle)

    return img, angles, rule_key


def _draw_hud(img, state, depth_pct, bar_y, rule_key, knee_angle, hip_angle):
    h, w = img.shape[:2]

    # Semi-transparent top banner
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # Rep count
    cv2.putText(img, "REPS", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    cv2.putText(img, str(int(state["count"])), (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 136), 3, cv2.LINE_AA)

    # Knee angle label
    cv2.putText(img, f"Knee: {int(knee_angle)}°", (120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f"Hip:  {int(hip_angle)}°", (120, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2, cv2.LINE_AA)

    # Depth percentage text near bar
    cv2.putText(img, f"DEPTH", (w - 90, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(img, f"{int(depth_pct)}%", (w - 88, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 210, 255), 2, cv2.LINE_AA)

    # Depth bar (right edge)
    bar_x = w - 30
    rule = FEEDBACK_RULES.get(rule_key) or MATH_RULES.get(rule_key, {})
    level = rule.get("level", "neutral")
    bar_color = {"good": (0, 255, 136), "bad": (0, 0, 255), "warn": (0, 165, 255)}.get(level, (100, 100, 100))

    cv2.rectangle(img, (bar_x, 150), (bar_x + 18, 480), (40, 40, 40), -1)    # track
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + 18, 480), bar_color, -1)      # fill
    cv2.rectangle(img, (bar_x, 150), (bar_x + 18, 480), (80, 80, 80), 1)      # border

    # Status label bottom-left
    status_text = (FEEDBACK_RULES.get(rule_key) or MATH_RULES.get(rule_key, {})).get("status", "")
    label_bg = {"good": (0, 80, 40), "bad": (80, 0, 0), "warn": (80, 60, 0)}.get(level, (30, 30, 30))
    label_color = {"good": (0, 255, 136), "bad": (80, 80, 255), "warn": (0, 200, 255)}.get(level, (200, 200, 200))
    cv2.rectangle(img, (0, h - 55), (w - 40, h), label_bg, -1)
    cv2.putText(img, status_text, (12, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, label_color, 2, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────────────────────────
def sidebar_ui():
    st.sidebar.markdown("## 🏋️ FormFlex Settings")
    source = st.sidebar.radio("Input Source", ("📁 Video Upload", "📷 Webcam"), label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Status**")
    if ml_model:
        st.sidebar.success("✅ RF Model Loaded")
        classes = list(FEEDBACK_RULES.keys())
        st.sidebar.caption(f"Classes: {', '.join(c.replace('s_','') for c in classes)}")
    else:
        st.sidebar.error("❌ squat.pkl not found — run train.py")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**What We Track**")
    st.sidebar.caption(
        "• 33 MediaPipe landmarks\n"
        "• 11 joint angles (atan2)\n"
        "• Knee / Hip / Ankle / Spine\n"
        "• Rep counter (state machine)\n"
        "• ML classification (Random Forest)\n"
        "• Audio alerts (Pygame)"
    )
    return source


def main():
    # ── Header
    st.markdown("""
    <h1 style='text-align:center;background:linear-gradient(90deg,#00d4ff,#00ff88);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    font-size:2.4rem;font-weight:800;margin-bottom:0'>
    🏋️ FormFlex — AI Squat Analyzer
    </h1>
    <p style='text-align:center;color:#888;font-size:0.92rem;margin-top:4px'>
    33-point MediaPipe tracking · 11 joint angles · Random Forest classification · Multi-modal feedback
    </p>
    <hr style='border-color:#222;margin:10px 0 20px'>
    """, unsafe_allow_html=True)

    source = sidebar_ui()

    col_vid, col_stats = st.columns([3, 1])

    with col_vid:
        stframe = st.empty()

    with col_stats:
        st.markdown("#### 📊 Live Stats")
        kpi_reps    = st.empty()
        kpi_knee    = st.empty()
        kpi_hip     = st.empty()
        kpi_depth   = st.empty()
        st.markdown("---")
        st.markdown("#### 🗒 Feedback")
        feedback_box = st.empty()
        tip_box      = st.empty()
        st.markdown("---")
        st.markdown("#### 📐 Joint Angles")
        angle_table  = st.empty()

    # ── Persistent state via session_state (survives Streamlit reruns)
    if "count" not in st.session_state:
        st.session_state["count"] = 0
    if "dir" not in st.session_state:
        st.session_state["dir"] = 0

    # Pass session_state directly — it behaves like a dict for key access
    state = st.session_state

    def render_stats(angles, rule_key, state):
        reps = int(state["count"])
        knee = angles.get("R Knee", 0)
        hip  = angles.get("R Hip", 0)
        depth = float(np.interp(knee, (70, 160), (100, 0)))
        depth = max(0.0, min(100.0, depth))

        kpi_reps.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Reps</div>
          <div class="kpi-value">{reps}</div>
        </div>""", unsafe_allow_html=True)

        knee_cls = "bad" if knee < 70 else ("good" if 90 <= knee <= 130 else "")
        kpi_knee.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Knee Angle</div>
          <div class="kpi-value {knee_cls}">{int(knee)}°</div>
        </div>""", unsafe_allow_html=True)

        hip_cls = "bad" if hip < 45 else ""
        kpi_hip.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Hip Angle</div>
          <div class="kpi-value {hip_cls}">{int(hip)}°</div>
        </div>""", unsafe_allow_html=True)

        kpi_depth.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Squat Depth</div>
          <div class="kpi-value">{int(depth)}%</div>
        </div>""", unsafe_allow_html=True)

        # Feedback
        rule = FEEDBACK_RULES.get(rule_key) or MATH_RULES.get(rule_key, {})
        level = rule.get("level", "neutral")
        status = rule.get("status", "—")
        tip    = rule.get("tip", "")
        fb_cls = {"good":"feedback-good","bad":"feedback-bad","warn":"feedback-warn"}.get(level,"feedback-neutral")
        feedback_box.markdown(f'<div class="feedback-box {fb_cls}">{status}</div>', unsafe_allow_html=True)
        tip_box.markdown(f'<div class="correction-tip">💡 {tip}</div>', unsafe_allow_html=True)

        # Angle table
        rows = "".join(
            f'<div class="angle-row"><span class="angle-name">{n}</span>'
            f'<span class="angle-val">{v}°</span></div>'
            for n, v in angles.items()
        )
        angle_table.markdown(f'<div>{rows}</div>', unsafe_allow_html=True)

    # ── VIDEO UPLOAD
    if "Video" in source:
        st.sidebar.markdown("**Upload a side-view squat video:**")
        uploaded = st.sidebar.file_uploader("", type=["mp4", "mov", "avi"])
        if uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            delay = 1.0 / fps
            running = st.sidebar.checkbox("▶ Play / Pause", value=True)
            while running:
                ok, frame = cap.read()
                if not ok:
                    st.sidebar.success("Video finished.")
                    break
                out, angles, rule_key = process_frame(frame, state)
                stframe.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                              channels="RGB", use_container_width=True)
                render_stats(angles, rule_key, state)
                time.sleep(delay)
            cap.release()

    # ── WEBCAM
    elif "Webcam" in source:
        st.sidebar.info("Stand **sideways** to the camera so both legs are visible.")
        run = st.sidebar.checkbox("▶ Start Camera")
        if run:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open webcam.")
            while run:
                ok, frame = cap.read()
                if not ok:
                    st.error("Webcam read error.")
                    break
                out, angles, rule_key = process_frame(frame, state)
                stframe.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                              channels="RGB", use_container_width=True)
                render_stats(angles, rule_key, state)
            cap.release()


if __name__ == "__main__":
    main()
