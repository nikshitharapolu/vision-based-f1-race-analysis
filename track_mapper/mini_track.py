"""
track_mapper/mini_track.py
===========================
Projects car centroids through the homography matrix onto a top-down
circuit map and renders a mini-map overlay.

Analogous to mini_court/ in abdullahtarek/tennis_analysis.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

# ── Circuit map configs ───────────────────────────────────────────────────────

CIRCUIT_MAPS: dict[str, dict] = {
    "silverstone": {
        "world_bbox":      (-50, -50, 450, 450),
        "pit_entry_line":  (0.0, 0.0, 12.0, 0.0),
        "pit_exit_line":   (0.0, 5.0, 12.0, 5.0),
    },
    "monaco": {
        "world_bbox":      (-30, -30, 280, 350),
        "pit_entry_line":  (0.0, 0.0,  9.0, 0.0),
        "pit_exit_line":   (0.0, 4.5,  9.0, 4.5),
    },
    "default": {
        "world_bbox":      (-50, -50, 450, 450),
        "pit_entry_line":  (0.0, 0.0, 12.0, 0.0),
        "pit_exit_line":   (0.0, 5.0, 12.0, 5.0),
    },
}

MINI_W, MINI_H  = 200, 200
MINI_MARGIN     = 10   # px from bottom-left of frame

# BGR colours per track_id mod 12
_PALETTE = [
    (0,255,255),(0,140,255),(0,255,128),(255,0,128),(255,255,0),
    (128,0,255),(0,80,255),(255,80,0),(200,200,200),(128,255,128),
    (255,128,255),(0,200,200),
]


@dataclass
class TrackCoord:
    """Car position in top-down world coordinates (metres)."""
    track_id:  int
    frame_idx: int
    x:         float
    y:         float
    speed_kmh: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


class MiniTrack:
    """
    Maintains top-down car positions and renders a mini-map overlay.
    """

    def __init__(
        self,
        circuit:              str  = "default",
        keypoints_per_frame:  dict | None = None,
        size:                 tuple[int,int] = (MINI_W, MINI_H),
    ):
        self.circuit     = circuit
        self.map_cfg     = CIRCUIT_MAPS.get(circuit, CIRCUIT_MAPS["default"])
        self.kp_frames   = keypoints_per_frame or {}
        self.mini_w, self.mini_h = size
        self._bg         = self._build_bg() if _CV2 else None
        # Populated by project_cars() — kept for draw_mini_map access
        self.car_positions: dict[int, list[TrackCoord]] = {}

    # ── Projection ────────────────────────────────────────────────────────────

    def project_cars(
        self,
        car_tracks: list[dict[int, dict]],
    ) -> dict[int, list[TrackCoord]]:
        """
        Project every car centroid through the nearest homography.
        Returns {track_id: [TrackCoord, ...]}.
        """
        result: dict[int, list[TrackCoord]] = {}
        for fi, frame_dict in enumerate(car_tracks):
            H = self._get_H(fi)
            for tid, det in frame_dict.items():
                bbox = det["bbox"]
                cx   = (bbox[0] + bbox[2]) / 2.0
                cy   = (bbox[1] + bbox[3]) / 2.0
                if H is not None and _CV2:
                    wx, wy = self._project(cx, cy, H)
                else:
                    wx, wy = cx, cy
                coord = TrackCoord(track_id=tid, frame_idx=fi, x=wx, y=wy)
                result.setdefault(tid, []).append(coord)
        self.car_positions = result
        return result

    def is_in_pit_lane(self, coord: TrackCoord) -> bool:
        """True if coord is within the pit-lane bounding strip."""
        entry = self.map_cfg.get("pit_entry_line")
        exit_ = self.map_cfg.get("pit_exit_line")
        if not entry or not exit_:
            return False
        min_y = min(entry[1], exit_[1])
        max_y = max(entry[3], exit_[3]) + 15.0
        min_x = min(entry[0], exit_[0]) - 2.0
        max_x = max(entry[2], exit_[2]) + 2.0
        return min_x <= coord.x <= max_x and min_y <= coord.y <= max_y

    # ── Rendering ─────────────────────────────────────────────────────────────

    def draw_mini_map(
        self,
        frame:           np.ndarray,
        frame_positions: dict[int, TrackCoord],
        car_speeds:      dict[int, list[float]] | None = None,
        frame_idx:       int = 0,
    ) -> np.ndarray:
        """Render the mini-map and paste it into the bottom-left of frame."""
        if not _CV2 or self._bg is None:
            return frame
        mini = self._bg.copy()
        for tid, coord in frame_positions.items():
            px, py = self._world_to_mini(coord.x, coord.y)
            col    = _PALETTE[tid % len(_PALETTE)]
            cv2.circle(mini, (int(px), int(py)), 4, col, -1)
            label  = str(tid)
            cv2.putText(mini, label, (int(px)+5, int(py)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, col, 1)
        # Paste into frame
        h, w = frame.shape[:2]
        x0   = MINI_MARGIN
        y0   = h - MINI_MARGIN - self.mini_h
        if y0 >= 0 and x0 + self.mini_w <= w:
            frame[y0:y0+self.mini_h, x0:x0+self.mini_w] = mini
        return frame

    def get_positions_at_frame(
        self,
        frame_idx: int,
    ) -> dict[int, TrackCoord]:
        """Return {track_id: TrackCoord} for all cars visible at frame_idx."""
        return {
            tid: next((c for c in coords if c.frame_idx == frame_idx), None)
            for tid, coords in self.car_positions.items()
            if any(c.frame_idx == frame_idx for c in coords)
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_H(self, frame_idx: int) -> np.ndarray | None:
        if not self.kp_frames:
            return None
        candidates = [
            (abs(fi - frame_idx), fi)
            for fi, r in self.kp_frames.items()
            if r.valid
        ]
        if not candidates:
            return None
        _, nearest = min(candidates)
        return self.kp_frames[nearest].homography

    @staticmethod
    def _project(cx: float, cy: float, H: np.ndarray) -> tuple[float, float]:
        pt  = np.array([[[cx, cy]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pt, H)
        return float(dst[0,0,0]), float(dst[0,0,1])

    def _world_to_mini(self, wx: float, wy: float) -> tuple[float, float]:
        x0, y0, x1, y1 = self.map_cfg["world_bbox"]
        px = (wx - x0) / (x1 - x0) * self.mini_w
        py = (wy - y0) / (y1 - y0) * self.mini_h
        return (
            float(np.clip(px, 0, self.mini_w - 1)),
            float(np.clip(py, 0, self.mini_h - 1)),
        )

    def _build_bg(self) -> np.ndarray:
        bg = np.zeros((self.mini_h, self.mini_w, 3), dtype=np.uint8)
        bg[:] = (20, 20, 30)
        cv2.rectangle(bg, (2,2), (self.mini_w-3, self.mini_h-3), (80,80,80), 1)
        entry = self.map_cfg.get("pit_entry_line")
        if entry:
            p0 = tuple(int(v) for v in self._world_to_mini(entry[0], entry[1]))
            p1 = tuple(int(v) for v in self._world_to_mini(entry[2], entry[3]))
            cv2.line(bg, p0, p1, (0,200,255), 1)
        return bg

    @staticmethod
    def car_color(track_id: int) -> tuple[int,int,int]:
        return _PALETTE[track_id % len(_PALETTE)]
