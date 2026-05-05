"""
utils/video_utils.py
=====================
Frame I/O, bounding box drawing, speed annotation, mini-map overlay,
leaderboard strip, and commentary text rendering.

Extends the tennis_analysis utils/video_utils.py with F1-specific
rendering: team-coloured boxes, speed labels, commentary ticker,
crash/flag event banners.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

UNIFIED_CLASSES = [
    "car","RedBull","Mercedes","Ferrari","McLaren","Alpine","AstonMartin",
    "Williams","Haas","KickSauber","RacingBulls",
    "track_surface","crash","penalty_car","pitstop","race_start","marshal",
    "yellow_flag","safety_car","off_track","on_track",
]

# BGR colours per class
CLASS_COLORS: dict[int, tuple] = {
    0:  (200,200,200), 1:  (255,30,30),   2:  (180,210,180),
    3:  (0,0,220),     4:  (0,140,255),   5:  (220,80,160),
    6:  (0,160,80),    7:  (200,170,255), 8:  (60,60,60),
    9:  (50,200,50),   10: (100,100,220), 11: (50,200,100),
    12: (0,0,255),     13: (0,50,255),    14: (255,200,0),
    15: (0,255,255),   16: (255,128,0),   17: (0,220,255),
    18: (255,255,255), 19: (200,100,50),  20: (50,200,200),
}

def _color(cls_id: int) -> tuple:
    return CLASS_COLORS.get(cls_id, (0,255,0))

def _name(cls_id: int) -> str:
    if 0 <= cls_id < len(UNIFIED_CLASSES):
        return UNIFIED_CLASSES[cls_id]
    return f"cls_{cls_id}"


# ══════════════════════════════════════════════════════════════════════════════
#  I/O
# ══════════════════════════════════════════════════════════════════════════════

def read_video(path: str) -> list[np.ndarray]:
    if not _CV2:
        raise ImportError("pip install opencv-python")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"  [Video] Read {len(frames)} frames from {path}")
    return frames


def get_video_fps(path: str) -> float:
    if not _CV2:
        return 30.0
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def save_video(
    frames:     list[np.ndarray],
    out_path:   str,
    fps:        float = 30.0,
    codec:      str   = "mp4v",
) -> None:
    if not _CV2:
        raise ImportError("pip install opencv-python")
    if not frames:
        return
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"  [Video] Saved {len(frames)} frames → {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
#  ANNOTATION RENDERER
# ══════════════════════════════════════════════════════════════════════════════
# Detect movement direction once for the whole video (more stable)

def draw_annotations(
    frames:          list[np.ndarray],
    car_tracks:      list[dict[int, dict]],
    car_speeds:      dict[int, list[float]],
    mini_track,
    leaderboard_state: dict,
    commentary_lines:  list[tuple[int, int, str]],
    events,
    show_minimap:    bool = False,
    race_direction:  str  = "auto",
) -> list[np.ndarray]:
    """
    Full annotation pipeline — returns annotated copies of all frames.
    show_minimap=False by default — pass True to show the top-down track map.
    """
    # Event type → chat dot colour (BGR)
    EVENT_COLORS = {
        "OVERTAKE":     (50,  200, 50),    # green
        "PIT_ENTRY":    (50,  200, 255),   # yellow
        "PIT_EXIT":     (50,  200, 255),   # yellow
        "CLOSE_BATTLE": (255, 150, 50),    # orange
        "CRASH":        (50,  50,  255),   # red
        "YELLOW_FLAG":  (0,   200, 255),   # yellow
        "RACE_START":   (255, 255, 50),    # bright yellow
        "DEFAULT":      (150, 150, 150),   # grey
    }

    # Pre-build the full chat message list from commentary_lines
    # Each entry: {frame_idx, text, timestamp, color}
    chat_history: list[dict] = []
    fps_est = 30.0   # used for timestamp display only
    COMMENTARY_DELAY_FRAMES = 15   # show commentary 0.5s after event detected


    for (f_start, f_end, text) in sorted(commentary_lines, key=lambda x: x[0]):
        # Delay commentary slightly so viewer sees event before reading about it
        display_start = f_start + COMMENTARY_DELAY_FRAMES
        secs = int(f_start / fps_est)
        ts   = f"{secs // 60}:{secs % 60:02d}"
        # Pick colour based on keywords in the text
        col = EVENT_COLORS["DEFAULT"]
        tl  = text.lower()
        if any(w in tl for w in ["passes", "overtake", "through", "move"]):
            col = EVENT_COLORS["OVERTAKE"]
        elif any(w in tl for w in ["pit", "box"]):
            col = EVENT_COLORS["PIT_ENTRY"]
        elif any(w in tl for w in ["crash", "incident", "safety car", "barriers"]):
            col = EVENT_COLORS["CRASH"]
        elif any(w in tl for w in ["yellow", "caution", "slow"]):
            col = EVENT_COLORS["YELLOW_FLAG"]
        elif any(w in tl for w in ["battle", "gearbox", "drs", "wheel"]):
            col = EVENT_COLORS["CLOSE_BATTLE"]
        elif any(w in tl for w in ["lights out", "racing", "underway"]):
            col = EVENT_COLORS["RACE_START"]

        chat_history.append({
            "frame_start": display_start,
            "text":        text,
            "timestamp":   ts,
            "color":       col,
        })

    # Detect movement direction once from middle of video (stable)
    mid_frame = len(frames) // 2
    direction = _detect_movement_direction(car_tracks, mid_frame, lookback=120)
    print(f"  [Renderer] Detected movement direction: {direction}")
   
    output = []
    n      = len(frames)

    for fi, frame in enumerate(frames):
        out = frame.copy()

        # 1. Car bounding boxes + speed labels
        if fi < len(car_tracks):
            out = _draw_car_boxes(out, car_tracks[fi], car_speeds, fi)

        # 2. Mini-map overlay (off by default)
        if show_minimap and mini_track is not None and hasattr(mini_track, "get_positions_at_frame"):
            positions = mini_track.get_positions_at_frame(fi)
            out = mini_track.draw_mini_map(out, positions, car_speeds, fi)
 
        # 3. Leaderboard strip
        lb = _nearest(leaderboard_state, fi)
        if lb:
            out = _draw_leaderboard(out, lb)

        # 4. Chat box — drawn onto sidebar (handled after loop)
        visible_msgs = [m for m in chat_history if m["frame_start"] <= fi]

        # 5. Event banner (crash, yellow flag)
        out = _draw_event_banner(out, events, fi)

        # 6. Frame counter
        _draw_frame_counter(out, fi, n)

        # Attach chat sidebar as a separate column next to the video
        # Build top-3 panel + chat sidebar stacked vertically
        sidebar_w   = 300
        frame_h     = out.shape[0]

        # Detect direction locally per frame — handles camera angle changes
        if race_direction == "auto":
            direction = _detect_movement_direction(car_tracks, fi, lookback=15)
        else:
            direction = race_direction

        top3_panel  = _build_top3_panel(
            car_tracks  = car_tracks,
            car_speeds  = car_speeds,
            frame_idx   = fi,
            width       = sidebar_w,
            leaderboard = leaderboard_state,
            direction   = direction,
        )
        chat_panel  = _build_chat_sidebar(
            visible_msgs,
            height = frame_h - top3_panel.shape[0],
            width  = sidebar_w,
            max_lines=10,
        )
        sidebar     = np.vstack([top3_panel, chat_panel])
        combined    = np.hstack([out, sidebar])
        output.append(combined)


    return output


def _draw_car_boxes(
    frame:      np.ndarray,
    frame_dict: dict[int, dict],
    car_speeds: dict[int, list[float]],
    fi:         int,
) -> np.ndarray:
    if not _CV2:
        return frame

    frame_labels = []   # track label positions this frame to avoid overlaps

    for tid, det in frame_dict.items():
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        cls_id   = det.get("class_id", 0)
        col      = _color(cls_id)
        interp   = det.get("conf", 1.0) == 0.0
        thickness= 1 if interp else 2

        cv2.rectangle(frame, (x1,y1), (x2,y2), col, thickness)

        spd_list = car_speeds.get(tid, [])
        spd      = spd_list[fi] if fi < len(spd_list) else 0.0
        cls_name = _name(cls_id) if cls_id > 0 else f"#{tid}"
        label    = f"{cls_id} {cls_name} {spd:.0f}km/h"
        _label_pill(frame, label, x1, y1, col, frame_labels)

    return frame


def _draw_leaderboard(frame: np.ndarray, lb) -> np.ndarray:
    if not _CV2 or not hasattr(lb, "entries") or not lb.entries:
        return frame
    h, w   = frame.shape[:2]
    x0     = w - 190
    y0     = 20
    line_h = 16
    panel_h= len(lb.entries) * line_h + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0-4, y0-4), (w-4, y0+panel_h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, entry in enumerate(lb.entries[:20]):
        y = y0 + i*line_h + line_h
        gap = entry.gap[:8] if len(entry.gap) > 8 else entry.gap
        cv2.putText(frame, f"P{entry.position:>2} {entry.driver_number:<4} {gap}",
                    (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255,255,255), 1, cv2.LINE_AA)
    if lb.lap:
        cv2.putText(frame, f"LAP {lb.lap}",
                    (x0, y0-8), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0,220,255), 1, cv2.LINE_AA)
    return frame


def _draw_commentary(frame: np.ndarray, text: str) -> np.ndarray:
    # Legacy single-line ticker — no longer used, kept for compatibility
    return frame

def _detect_movement_direction(
    car_tracks: list[dict[int, dict]],
    frame_idx:  int,
    lookback:   int = 15,   # short window = responds to camera cuts
) -> str:
    """
    Detect dominant movement direction around frame_idx using a short
    local window. Called per-frame so camera angle changes are handled.
    """
    f_now  = min(frame_idx, len(car_tracks) - 1)
    f_prev = max(0, f_now - lookback)

    if f_prev == f_now:
        return "right"

    now_dict  = car_tracks[f_now]
    prev_dict = car_tracks[f_prev]
    common    = set(now_dict.keys()) & set(prev_dict.keys())

    if not common:
        return "right"

    dx_total = 0.0
    dy_total = 0.0
    count    = 0

    for tid in common:
        b_now  = now_dict[tid]["bbox"]
        b_prev = prev_dict[tid]["bbox"]
        cx_now  = (b_now[0]  + b_now[2])  / 2
        cy_now  = (b_now[1]  + b_now[3])  / 2
        cx_prev = (b_prev[0] + b_prev[2]) / 2
        cy_prev = (b_prev[1] + b_prev[3]) / 2
        dx_total += cx_now - cx_prev
        dy_total += cy_now - cy_prev
        count    += 1

    if count == 0:
        return "right"

    dx_avg = dx_total / count
    dy_avg = dy_total / count

    # Ignore tiny jitter — need meaningful movement
    if abs(dx_avg) < 0.3 and abs(dy_avg) < 0.3:
        return "right"

    if abs(dx_avg) > abs(dy_avg):
        return "left" if dx_avg < 0 else "right"
    else:
        return "up" if dy_avg < 0 else "down"

def _rank_key(bbox: list[float], direction: str) -> float:
    """
    Return a sort key for a bounding box such that the leading car
    gets the lowest key value (sorted ascending = P1 first).

    Leading car definition per direction:
        right → highest X  (negate so lowest key = rightmost)
        left  → lowest  X
        up    → lowest  Y  (highest on screen = furthest ahead)
        down  → highest Y  (negate so lowest key = lowest on screen)
    """
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2

    if direction == "right":
        return -cx        # rightmost = leading
    elif direction == "left":
        return cx         # leftmost  = leading
    elif direction == "down":
        return -cy        # lowest on screen = leading
    else:                 # "up" — default
        return cy         # highest on screen (smallest Y) = leading

def _build_top3_panel(
    car_tracks:  list[dict[int, dict]],
    car_speeds:  dict[int, list[float]],
    frame_idx:   int,
    width:       int = 300,
    leaderboard: dict = None,
    direction:   str = "right",
) -> np.ndarray:
    """
    Build a top-3 leaderboard panel showing current race positions.
    Uses running order from Y-position of tracked cars, or OCR leaderboard if available.
    """
    panel_h = 130
    panel   = np.zeros((panel_h, width, 3), dtype=np.uint8)
    panel[:] = (22, 22, 22)

    font   = cv2.FONT_HERSHEY_SIMPLEX
    pad    = 10

    # Header
    cv2.rectangle(panel, (0, 0), (width, 28), (35, 35, 35), -1)
    cv2.line(panel, (0, 28), (width, 28), (60, 60, 60), 1)
    # Trophy dots
    trophy_colors = [(0, 215, 255), (0, 185, 192), (0, 140, 105)]  # gold, silver, bronze
    for rank, col in enumerate(trophy_colors):
        cv2.circle(panel, (pad + rank * 12, 14), 4, col, -1)
    cv2.putText(panel, "TOP 3 POSITIONS", (pad + 44, 19),
                font, 0.36, (200, 200, 200), 1, cv2.LINE_AA)

    # Get current frame's track data
    frame_dict = car_tracks[frame_idx] if frame_idx < len(car_tracks) else {}

    # Try OCR leaderboard first (most accurate)
    top3_entries = []
    if leaderboard:
        nearest_fi = min(leaderboard.keys(), key=lambda k: abs(k - frame_idx), default=None)
        if nearest_fi is not None:
            lb = leaderboard[nearest_fi]
            if hasattr(lb, "entries") and lb.entries:
                for entry in lb.entries[:3]:
                    spd = 0.0
                    # Try to match driver number to a tracker ID for speed
                    for tid, det in frame_dict.items():
                        s = car_speeds.get(tid, [])
                        if frame_idx < len(s) and s[frame_idx] > spd:
                            spd = s[frame_idx]
                    top3_entries.append({
                        "pos":    entry.position,
                        "label":  f"#{entry.driver_number}",
                        "gap":    entry.gap,
                        "speed":  spd,
                        "cls_id": -1,
                    })

    # Fallback: rank by position of bounding box centroids
    # Auto-detect movement direction from recent frames
    if not top3_entries and frame_dict:
        ranked = sorted(
            frame_dict.items(),
            key=lambda kv: _rank_key(kv[1]["bbox"], direction)
        )
        for rank, (tid, det) in enumerate(ranked[:3]):
            s     = car_speeds.get(tid, [])
            spd   = s[frame_idx] if frame_idx < len(s) else 0.0
            cls_id= det.get("class_id", 0)
            label = f"#{tid}"
            if 0 <= cls_id < len(UNIFIED_CLASSES):
                label = UNIFIED_CLASSES[cls_id] if cls_id > 0 else f"#{tid}"
            top3_entries.append({
                "pos":    rank + 1,
                "label":  label,
                "gap":    "LEADER" if rank == 0 else "",
                "speed":  spd,
                "cls_id": cls_id,
            })

    if not top3_entries:
        cv2.putText(panel, "Detecting positions...", (pad, 65),
                    font, 0.36, (70, 70, 70), 1, cv2.LINE_AA)
        return panel

    # Draw each position row
    rank_colors = [
        (0,  215, 255),   # P1 — gold
        (0,  192, 192),   # P2 — silver
        (0,  140, 105),   # P3 — bronze
    ]
    row_h = 30
    for i, entry in enumerate(top3_entries[:3]):
        y    = 28 + 6 + i * row_h
        col  = rank_colors[i]

        # Position badge
        cv2.rectangle(panel, (pad, y), (pad + 22, y + 22), col, -1)
        cv2.putText(panel, f"P{entry['pos']}", (pad + 2, y + 15),
                    font, 0.38, (15, 15, 15), 1, cv2.LINE_AA)

        # Car label
        cv2.putText(panel, entry["label"], (pad + 30, y + 15),
                    font, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

        # Speed
        if entry["speed"] > 1.0:
            spd_str = f"{entry['speed']:.0f}km/h"
            cv2.putText(panel, spd_str, (width - pad - 70, y + 15),
                        font, 0.36, (140, 140, 140), 1, cv2.LINE_AA)

        # Gap (P2, P3 only)
        if i > 0 and entry.get("gap"):
            gap_str = str(entry["gap"])[:8]
            cv2.putText(panel, gap_str, (width - pad - 70, y + 15),
                        font, 0.34, (100, 180, 100), 1, cv2.LINE_AA)

    # Bottom divider
    cv2.line(panel, (0, panel_h - 1), (width, panel_h - 1), (50, 50, 50), 1)

    return panel

def _build_chat_sidebar(
    messages:  list[dict],
    height:    int,
    width:     int  = 300,
    max_lines: int  = 20,
) -> np.ndarray:
    """
    Build a standalone chat sidebar image (height x width, BGR).
    Returned image is stacked horizontally next to the video frame —
    it does NOT overlay anything on the video itself.
    """
    panel     = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:]  = (18, 18, 18)   # dark background

    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_sm   = 0.34
    font_msg  = 0.38
    thick     = 1
    pad       = 8
    line_h    = 52
    max_chars = 24

    # ── Header ────────────────────────────────────────────────────────────────
    cv2.rectangle(panel, (0, 0), (width, 36), (30, 30, 30), -1)
    cv2.line(panel, (0, 36), (width, 36), (60, 60, 60), 1)
    # Chat icon dots
    for dx in [10, 16, 22]:
        cv2.circle(panel, (dx, 18), 3, (100, 180, 100), -1)
    cv2.putText(panel, "LIVE COMMENTARY", (32, 23),
                font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    if not messages:
        cv2.putText(panel, "Waiting for events...", (pad, 70),
                    font, font_sm, (80, 80, 80), 1, cv2.LINE_AA)
        return panel

    # ── Messages — show most recent, scrolled to bottom ───────────────────────
    visible   = messages[-max_lines:]
    y_start   = 36 + pad

    for i, msg in enumerate(visible):
        y_top = y_start + i * line_h

        if y_top + line_h > height - 24:
            break

        col  = msg.get("color", (130, 130, 130))
        ts   = msg.get("timestamp", "")
        text = msg.get("text", "")

        # Timestamp + coloured dot
        cv2.putText(panel, ts, (pad, y_top + 12),
                    font, font_sm, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.circle(panel, (pad + 32, y_top + 9), 4, col, -1)

        # Word-wrap text into the sidebar width
        max_chars = 24
        words     = text.split()
        lines     = []
        current   = ""
        for word in words:
            test=(current+" "+word).strip()
            if len(test) <= max_chars:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)

        for li, line in enumerate(lines[:3]):
            cv2.putText(panel, line,
                        (pad + 42, y_top + 12 + li * 15),
                        font, font_msg, (225, 225, 225), thick, cv2.LINE_AA)

        # Separator
        cv2.line(panel, (pad, y_top + line_h - 3),
                 (width - pad, y_top + line_h - 3),
                 (40, 40, 40), 1)

    # ── Bottom gradient bar ───────────────────────────────────────────────────
    cv2.rectangle(panel, (0, height - 22), (width, height), (25, 25, 25), -1)
    cv2.putText(panel, f"{len(messages)} events", (pad, height - 7),
                font, 0.32, (70, 70, 70), 1, cv2.LINE_AA)

    return panel


def _draw_event_banner(
    frame:  np.ndarray,
    events, fi: int,
) -> np.ndarray:
    if not _CV2:
        return frame
    from analysis.race_stats import EventType

    BANNER_CFG = {
        EventType.CRASH:       {"col": (0,0,200),   "label": "⚠ CRASH",       "sub": "Safety car may be deployed", "duration": 150},
        EventType.YELLOW_FLAG: {"col": (0,180,230),  "label": "⚠ YELLOW FLAG", "sub": "No overtaking — slow down",  "duration": 120},
    }

    for ev in sorted(events, key=lambda e: e.frame_idx, reverse=True):
        cfg = BANNER_CFG.get(ev.type)
        if cfg is None:
            continue

        # Only show AFTER event occurs, never before
        elapsed = fi - ev.frame_idx
        if not (0 <= elapsed <= cfg["duration"]):
            continue

        h, w   = frame.shape[:2]
        col    = cfg["col"]
        label  = cfg["label"]
        sub    = cfg["sub"]

        # Fade out gradually in last 30 frames
        alpha = 1.0
        if elapsed > cfg["duration"] - 30:
            alpha = (cfg["duration"] - elapsed) / 30.0

        # Banner background
        overlay = frame.copy()
        banner_h = 52
        y0 = h - banner_h - 50   # bottom of frame, above chat
        cv2.rectangle(overlay, (0, y0), (w - 300, y0 + banner_h), col, -1)
        cv2.addWeighted(overlay, alpha * 0.85, frame, 1 - alpha * 0.85, 0, frame)

        # Left coloured stripe
        cv2.rectangle(frame, (0, y0), (8, y0 + banner_h), (255,255,255), -1)

        # Main label
        cv2.putText(frame, label, (20, y0 + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85,
                    (255,255,255), 2, cv2.LINE_AA)

        # Sub label
        cv2.putText(frame, sub, (20, y0 + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (220,220,220), 1, cv2.LINE_AA)

        # Progress bar showing how long banner has been shown
        bar_w = int((w - 308) * (1 - elapsed / cfg["duration"]))
        cv2.rectangle(frame, (0, y0 + banner_h - 3),
                      (bar_w, y0 + banner_h), (255,255,255), -1)
        break

    return frame


def _draw_frame_counter(frame: np.ndarray, fi: int, total: int) -> None:
    if not _CV2:
        return
    cv2.putText(frame, f"{fi+1}/{total}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160,160,160), 1)


# Track label positions per frame to avoid overlaps
_label_positions: list[tuple[int,int,int,int]] = []

def _label_pill(frame, text, x1, y1, col, frame_labels=None):
    """Draw label pill, shifting position if it overlaps with existing labels."""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    fscale     = 0.4
    thick      = 1
    tw, th     = cv2.getTextSize(text, font, fscale, thick)[0]
    pad        = 3
    pill_w     = tw + pad * 2
    pill_h     = th + pad * 2 + 2

    # Default position: above box
    lx = x1
    ly = max(y1 - pill_h, 0)

    # Check for overlaps and shift down if needed
    if frame_labels is not None:
        attempts = 0
        while attempts < 5:
            overlap = False
            for (ox, oy, ow, oh) in frame_labels:
                # Check if current position overlaps with existing label
                if not (lx + pill_w < ox or lx > ox + ow or
                        ly + pill_h < oy or ly > oy + oh):
                    overlap = True
                    break
            if not overlap:
                break
            # Shift down by pill height + 2px margin
            ly += pill_h + 2
            # If gone below box bottom, shift right instead
            if ly > y1 + 60:
                ly = max(y1 - pill_h, 0)
                lx += pill_w + 4
            attempts += 1

        frame_labels.append((lx, ly, pill_w, pill_h))

    # Draw pill background
    cv2.rectangle(frame, (lx, ly), (lx + pill_w, ly + pill_h), (0,0,0), -1)
    cv2.putText(frame, text, (lx + pad, ly + pill_h - pad - 2),
                font, fscale, col, thick, cv2.LINE_AA)


def _nearest(d: dict, fi: int):
    if not d:
        return None
    return d[min(d.keys(), key=lambda k: abs(k-fi))]
