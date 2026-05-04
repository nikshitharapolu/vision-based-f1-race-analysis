"""
ocr/overlay_parser.py
======================
Extracts leaderboard and gap data from broadcast timing overlays
using Tesseract OCR.

Analogous to (new module) in the proposal — not present in tennis_analysis.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    import pytesseract
    _TESS = True
except ImportError:
    _TESS = False

# ── Broadcast region presets (relative to frame W/H) ─────────────────────────

REGIONS = {
    "sky_f1": {
        "timing_tower": (0.0, 0.05, 0.15, 0.95),
        "lap_counter":  (0.42, 0.02, 0.58, 0.08),
    },
    "f1tv": {
        "timing_tower": (0.0, 0.08, 0.18, 0.92),
        "lap_counter":  (0.40, 0.01, 0.60, 0.07),
    },
    "default": {
        "timing_tower": (0.0, 0.05, 0.20, 0.95),
        "lap_counter":  (0.40, 0.01, 0.60, 0.08),
    },
}

PSM_COLUMN = "--psm 4 --oem 1"
PSM_LINE   = "--psm 7 --oem 1"


@dataclass
class LeaderboardEntry:
    position:      int
    driver_number: str    # e.g. "44", "VER", "1"
    gap:           str    # e.g. "+1.234", "LAP 1", "LEADER"


@dataclass
class FrameLeaderboard:
    frame_idx: int
    lap:       int | None
    entries:   list[LeaderboardEntry] = field(default_factory=list)

    def number_to_position(self) -> dict[str, int]:
        return {e.driver_number: e.position for e in self.entries}

    def position_to_number(self) -> dict[int, str]:
        return {e.position: e.driver_number for e in self.entries}


class OverlayParser:
    """
    Extracts structured leaderboard state from broadcast frames.

    If Tesseract is not installed, falls back to mock data.
    """

    _ENTRY_RE = re.compile(
        r"(?:P)?(\d{1,2})\s+([A-Z0-9]{1,4})\s+([\+\-]?[\d\.]+|LAP\s*\d+|LEADER)",
        re.IGNORECASE,
    )
    _LAP_RE   = re.compile(r"LAP\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)

    def __init__(self, broadcast: str = "default"):
        self.regions     = REGIONS.get(broadcast, REGIONS["default"])
        self._last_valid: FrameLeaderboard | None = None

    def parse_frames(
        self,
        frames: list[np.ndarray],
        stride: int = 30,
    ) -> dict[int, FrameLeaderboard]:
        results: dict[int, FrameLeaderboard] = {}
        for fi in range(0, len(frames), stride):
            lb = self.parse_frame(frames[fi], frame_idx=fi)
            if lb and lb.entries:
                self._last_valid = lb
            results[fi] = lb or self._last_valid or self._mock(fi)
        return results

    def parse_frame(
        self, frame: np.ndarray, frame_idx: int = 0
    ) -> FrameLeaderboard | None:
        if not _TESS or not _CV2:
            return self._mock(frame_idx)
        try:
            crop = self._crop(frame, self.regions["timing_tower"])
            proc = self._preprocess(crop)
            text = pytesseract.image_to_string(proc, config=PSM_COLUMN)
            entries = self._parse_tower(text)
            lap     = self._parse_lap(frame)
            return FrameLeaderboard(frame_idx=frame_idx, lap=lap, entries=entries)
        except Exception as e:
            return None

    def _crop(self, frame: np.ndarray, region: tuple) -> np.ndarray:
        h, w = frame.shape[:2]
        x0 = int(region[0] * w); y0 = int(region[1] * h)
        x1 = int(region[2] * w); y1 = int(region[3] * h)
        return frame[y0:y1, x0:x1]

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        eq    = cv2.equalizeHist(gray)
        _, bw = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        h, w  = bw.shape
        return cv2.resize(bw, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

    def _parse_tower(self, text: str) -> list[LeaderboardEntry]:
        entries = []
        for m in self._ENTRY_RE.finditer(text):
            entries.append(LeaderboardEntry(
                position      = int(m.group(1)),
                driver_number = m.group(2).upper(),
                gap           = m.group(3).strip(),
            ))
        return entries

    def _parse_lap(self, frame: np.ndarray) -> int | None:
        if not _CV2 or not _TESS:
            return None
        try:
            crop = self._crop(frame, self.regions["lap_counter"])
            proc = self._preprocess(crop)
            text = pytesseract.image_to_string(proc, config=PSM_LINE)
            m    = self._LAP_RE.search(text)
            return int(m.group(1)) if m else None
        except Exception:
            return None

    @staticmethod
    def _mock(frame_idx: int) -> FrameLeaderboard:
        return FrameLeaderboard(
            frame_idx = frame_idx,
            lap       = None,
            entries   = [
                LeaderboardEntry(pos, num, f"+{pos*1.5:.3f}")
                for pos, num in enumerate(["1","44","16","63","55","4","14","23","18","10"], 1)
            ],
        )
