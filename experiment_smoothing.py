"""
Command-stability measurement (validation)

Requires the original SORT implementation (sort.py) on the path for the baseline.

Run:
    python experiment_smoothing.py
"""

import numpy as np
from collections import defaultdict, Counter

import cv2
from ultralytics import YOLO

# ---------------- config ----------------
VIDEO_PATH = r"C:\Users\gonth\Downloads\Video_Generation_From_Color_Boxes.mp4"
MODEL_PATH = r"C:\Users\gonth\Downloads\best.pt"
CONF = 0.35
COMMANDS = {0: "LEFT", 1: "RIGHT", 2: "FORWARD"}   # blue, red, white


def command_from_class(cls):
    return COMMANDS.get(int(cls), "NONE")


def metrics(seq_by_id):
    """seq_by_id: id -> list of commands in frame order.
    Returns (total flips, total switches, mean purity)."""
    flips = switches = 0
    purities = []
    for cmds in seq_by_id.values():
        if not cmds:
            continue
        flips += sum(1 for i in range(1, len(cmds)) if cmds[i] != cmds[i - 1])
        switches += len(set(cmds)) - 1
        top = Counter(cmds).most_common(1)[0][1]
        purities.append(top / len(cmds))
    mean_purity = float(np.mean(purities)) if purities else 0.0
    return flips, switches, mean_purity


# ---------------- baseline: original SORT + index assignment ----------------
def run_original(model):
    """The ORIGINAL pipeline: SORT tracker, class matched by list index.
    This is the defective version — kept to measure the defect."""
    from sort import Sort
    tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.3)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    seq_by_id = defaultdict(list)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF, verbose=False)[0]
        detections, classes = [], []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append([x1, y1, x2, y2, float(box.conf[0])])
            classes.append(int(box.cls[0]))

        detections = np.array(detections) if detections else np.empty((0, 5))
        tracks = tracker.update(detections)

        # ORIGINAL buggy index matching — measured on purpose
        for i, t in enumerate(tracks):
            tid = int(t[4])
            cls = classes[i] if i < len(classes) else -1
            seq_by_id[tid].append(command_from_class(cls))

    cap.release()
    return seq_by_id


# ---------------- fixed: tracker-native class association ----------------
def run_fixed(model):
    """The CORRECTED pipeline: model.track carries class WITH identity."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    seq_by_id = defaultdict(list)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        r = model.track(frame, persist=True, conf=CONF,
                        tracker="botsort.yaml", verbose=False)[0]
        b = r.boxes
        if b is None or b.id is None:
            continue
        ids = b.id.int().cpu().numpy()
        clss = b.cls.int().cpu().numpy()
        for tid, cls in zip(ids, clss):
            seq_by_id[int(tid)].append(command_from_class(cls))

    cap.release()
    return seq_by_id


def main():
    model = YOLO(MODEL_PATH)

    print("Running ORIGINAL pipeline (SORT + index assignment) ...")
    try:
        orig = run_original(model)
        f_o, s_o, p_o = metrics(orig)
    except ImportError:
        print("  [skipped] sort.py not found — baseline needs the original SORT module.")
        f_o = s_o = None; p_o = None

    print("Running FIXED pipeline (tracker-native association) ...")
    fixed = run_fixed(model)
    f_f, s_f, p_f = metrics(fixed)

    # ---------------- table ----------------
    print("\n" + "=" * 62)
    print(f"{'Pipeline':<40}{'flips':>7}{'switch':>7}{'purity':>8}")
    print("-" * 62)
    if f_o is not None:
        print(f"{'SORT + index assignment (original)':<40}{f_o:>7}{s_o:>7}{p_o:>8.3f}")
    else:
        print(f"{'SORT + index assignment (original)':<40}{'  n/a':>7}{'  n/a':>7}{'   n/a':>8}")
    print(f"{'Tracker-native association (fixed)':<40}{f_f:>7}{s_f:>7}{p_f:>8.3f}")
    print("=" * 62)

    if f_o:
        red = 100 * (f_o - f_f) / f_o
        print(f"\nFlip reduction: {red:.1f}%  ({f_o} -> {f_f})")
    print("\npurity = mean fraction of a track's frames on its dominant command "
          "(1.000 = perfectly stable).")


if __name__ == "__main__":
    main()