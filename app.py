"""
CONVEYOR COMMAND VISION  —  live YOLO + tracking control deck
=============================================================

Run:
    pip install streamlit ultralytics opencv-python-headless numpy
"""

import time
from collections import deque, Counter
import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ------------------------------------------------------------------ config
MODEL_PATH = "C:\\Users\gonth\\Downloads\\best.pt"          # <- your trained weights
CLASS_NAMES = {0: "blue", 1: "red", 2: "white"}
COMMANDS = {0: "LEFT", 1: "RIGHT", 2: "FORWARD"}

# command -> BGR box color (novelty: color == meaning)
CMD_COLOR = {
    "LEFT":    (255, 170,  0),   # amber
    "RIGHT":   (0,   140, 255),  # orange
    "FORWARD": (60,  220,  60),  # green
    "NONE":    (150, 150, 150),
}
CMD_ANGLE = {"LEFT": -90, "RIGHT": 90, "FORWARD": 0, "NONE": 0}
SMOOTH_WINDOW = 8               # frames of per-track voting


def command_from_class(cls):
    return COMMANDS.get(int(cls), "NONE")


# ------------------------------------------------------------------ page
st.set_page_config(page_title="Conveyor Command Vision",
                   page_icon="🧭", layout="wide")

st.markdown("""
<style>
  .stApp { background: radial-gradient(1200px 600px at 20% -10%, #1b2440 0%, #0b0f1a 55%); }
  h1, h2, h3, p, label, span { color: #e8ecf5 !important; }
  .badge { display:inline-block; padding:6px 14px; border-radius:999px;
           font-weight:700; letter-spacing:.5px; margin-right:8px;
           background:#151d33; border:1px solid #2a3a63; }
  .mono  { font-family: ui-monospace, Menlo, monospace; }
  .card  { background:#0f1626cc; border:1px solid #223055; border-radius:16px;
           padding:16px 18px; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🧭 Conveyor Command Vision")
st.markdown(
    '<span class="badge">YOLO11</span>'
    '<span class="badge">BoT-SORT tracking</span>'
    '<span class="badge">live command HUD</span>',
    unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Control")
    source = st.radio("Source", ["Upload video", "Live webcam"])
    conf = st.slider("Confidence", 0.10, 0.90, 0.35, 0.05)
    model_path = st.text_input("Weights path", MODEL_PATH)
    show_ids = st.checkbox("Show track IDs", True)
    up_file = None
    cam_index = 0
    if source == "Upload video":
        up_file = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"])
    else:
        cam_index = st.number_input("Camera index", 0, 8, 0, 1)
        st.caption("Local runs only. Cloud webcam needs streamlit-webrtc.")
    run = st.button("▶ Deploy", use_container_width=True, type="primary")


@st.cache_resource(show_spinner="Loading model…")
def load_model(path):
    return YOLO(path)


def steering_compass(angle_deg, label, size=180):
    """Return a BGR image of a needle rotated to angle_deg."""
    img = np.zeros((size, size, 3), np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), c - 8, (90, 110, 160), 2)
    for a in (-90, 0, 90):
        rad = np.deg2rad(a - 90)
        x = int(c + (c - 16) * np.cos(rad)); y = int(c + (c - 16) * np.sin(rad))
        cv2.circle(img, (x, y), 3, (120, 140, 190), -1)
    rad = np.deg2rad(angle_deg - 90)
    tip = (int(c + (c - 26) * np.cos(rad)), int(c + (c - 26) * np.sin(rad)))
    color = CMD_COLOR.get(label, (200, 200, 200))
    cv2.line(img, (c, c), tip, color, 4)
    cv2.circle(img, (c, c), 6, color, -1)
    cv2.putText(img, label, (10, size - 12), 0, 0.7, color, 2)
    return img


# ------------------------------------------------------------------ pipeline
def run_stream(cap, model, conf, show_ids):
    col_v, col_h = st.columns([3, 1])
    frame_slot = col_v.empty()
    comp_slot = col_h.empty()
    stat_slot = col_h.empty()
    log_slot = st.empty()

    history = {}                       # track_id -> deque of class votes
    frame_id = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # built-in tracker: cls + id already aligned, no manual SORT
        results = model.track(frame, persist=True, conf=conf,
                              tracker="botsort.yaml", verbose=False)[0]

        votes = Counter()
        boxes = results.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().numpy()
            clss = boxes.cls.int().cpu().numpy()

            for (x1, y1, x2, y2), tid, cls in zip(xyxy, ids, clss):
                tid = int(tid)
                # smoothing: vote over last N frames for this track
                history.setdefault(tid, deque(maxlen=SMOOTH_WINDOW)).append(int(cls))
                voted_cls = Counter(history[tid]).most_common(1)[0][0]
                cmd = command_from_class(voted_cls)
                votes[cmd] += 1

                color = CMD_COLOR[cmd]
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(frame, p1, p2, color, 2)
                tag = f"ID {tid} · {cmd}" if show_ids else cmd
                cv2.rectangle(frame, (p1[0], p1[1] - 22),
                              (p1[0] + 11 * len(tag), p1[1]), color, -1)
                cv2.putText(frame, tag, (p1[0] + 3, p1[1] - 6),
                            0, 0.55, (15, 15, 20), 2)

        majority = votes.most_common(1)[0][0] if votes else "NONE"
        fps = frame_id / (time.time() - t0 + 1e-9)

        frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                         channels="RGB", use_container_width=True)
        comp = steering_compass(CMD_ANGLE[majority], majority)
        comp_slot.image(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB),
                        caption="fleet command", use_container_width=True)
        stat_slot.markdown(
            f'<div class="card mono">frame <b>{frame_id}</b><br>'
            f'fps <b>{fps:4.1f}</b><br>tracks <b>{sum(votes.values())}</b></div>',
            unsafe_allow_html=True)
        if votes:
            log_slot.markdown(
                '<div class="card mono">'
                + " &nbsp; ".join(f'{k}:{v}' for k, v in votes.items())
                + '</div>', unsafe_allow_html=True)

    cap.release()


# ------------------------------------------------------------------ launch
if run:
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Could not load model at '{model_path}': {e}")
        st.stop()

    if source == "Upload video":
        if up_file is None:
            st.warning("Upload a video first.")
            st.stop()
        tmp = os.path.join(tempfile.gettempdir(), f"_in_{int(time.time())}.mp4")
        with open(tmp, "wb") as f:
            f.write(up_file.read())
        cap = cv2.VideoCapture(tmp)
    else:
        cap = cv2.VideoCapture(int(cam_index))

    if not cap.isOpened():
        st.error("Could not open the video source.")
        st.stop()

    run_stream(cap, model, conf, show_ids)
    st.success("Stream finished.")
else:
    st.info("Pick a source in the sidebar and hit **Deploy**.")