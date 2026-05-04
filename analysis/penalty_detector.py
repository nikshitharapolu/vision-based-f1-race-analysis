"""
analysis/penalty_detector.py
=============================
Detects racing-style penalties by analysing car trajectories,
relative positions, and flag states over multi-frame windows.

Penalty types (per proposal):
  TRACK_LIMIT_VIOLATION — all 4 wheels outside white line
  PUSHING_OFF_TRACK     — side-by-side → one car deviates off track
  UNFAIR_OVERTAKE       — overtake gained while off-track or under yellow
  WEAVING               — more than one lateral direction change defending
  PIT_LANE_SPEEDING     — speed > threshold while in pit zone
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class PenaltyType(Enum):
    TRACK_LIMIT_VIOLATION = auto()
    PUSHING_OFF_TRACK     = auto()
    UNFAIR_OVERTAKE       = auto()
    WEAVING               = auto()
    PIT_LANE_SPEEDING     = auto()


@dataclass
class PenaltyEvent:
    type:      PenaltyType
    frame_idx: int
    car_id:    int
    details:   dict[str, Any] = field(default_factory=dict)
    severity:  str = "investigation"   # "investigation" | "penalty" | "warning"

    def __str__(self) -> str:
        return (f"{self.type.name:<26}  Car {self.car_id}"
                f"  [{self.severity}]  frame={self.frame_idx}")


# ── Constants ─────────────────────────────────────────────────────────────────

PIT_SPEED_LIMIT_KMH      = 80.0    # standard F1 pit lane speed limit
TRACK_LIMIT_MARGIN_M     = 1.5     # allowance beyond track edge (metres)
WEAVE_ANGLE_THRESHOLD    = 30.0    # degrees: direction reversal counts as weave
WEAVE_WINDOW_FRAMES      = 60      # frames to look for multiple direction changes
PUSH_PROXIMITY_M         = 4.0     # cars must be this close to register a push
PUSH_DEVIATION_M         = 3.0     # off-track deviation needed to flag a push
YELLOW_FLAG_CLASS_ID     = 17      # per UNIFIED_CLASSES
OFF_TRACK_CLASS_ID       = 19


class PenaltyDetector:
    """
    Detects racing-style penalties from track positions, speeds, and YOLO events.

    Takes the output of RaceStats.compute_speeds() and
    MiniTrack.car_positions as input.
    """

    def __init__(
        self,
        car_positions:   dict[int, list],    # tid → list[TrackCoord]
        car_speeds:      dict[int, list[float]],
        yolo_detections: list[dict] | None = None,
        mini_track=      None,
        fps:             float = 30.0,
    ):
        self.car_positions   = car_positions
        self.car_speeds      = car_speeds
        self.yolo_dets       = yolo_dets = yolo_detections or []
        self.mini_track      = mini_track
        self.fps             = fps
        self._n_frames       = max(
            (max(c.frame_idx for c in coords) for coords in car_positions.values()
             if coords), default=0
        ) + 1

        # Pre-index YOLO detections by frame
        self._yolo_by_frame: dict[int, list[dict]] = {}
        for d in self.yolo_dets:
            fi = d.get("frame_idx", 0)
            self._yolo_by_frame.setdefault(fi, []).append(d)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_all(self) -> list[PenaltyEvent]:
        events: list[PenaltyEvent] = []
        events += self.detect_pit_speeding()
        events += self.detect_weaving()
        events += self.detect_pushing_off_track()
        events += self.detect_unfair_overtakes()
        events += self.detect_track_limit_violations()
        events.sort(key=lambda e: e.frame_idx)
        return events

    # ── Pit lane speeding ─────────────────────────────────────────────────────

    def detect_pit_speeding(self) -> list[PenaltyEvent]:
        events = []
        if self.mini_track is None:
            return events
        for tid, coords in self.car_positions.items():
            spd = self.car_speeds.get(tid, [])
            for c in coords:
                if not self.mini_track.is_in_pit_lane(c):
                    continue
                s = spd[c.frame_idx] if c.frame_idx < len(spd) else 0.0
                if s > PIT_SPEED_LIMIT_KMH:
                    events.append(PenaltyEvent(
                        type      = PenaltyType.PIT_LANE_SPEEDING,
                        frame_idx = c.frame_idx,
                        car_id    = tid,
                        details   = {"speed_kmh": round(s,1),
                                     "limit_kmh": PIT_SPEED_LIMIT_KMH},
                        severity  = "penalty",
                    ))
        return self._deduplicate(events, cooldown=60)

    # ── Weaving / defending ───────────────────────────────────────────────────

    def detect_weaving(self) -> list[PenaltyEvent]:
        events = []
        for tid, coords in self.car_positions.items():
            frame_to_x = {c.frame_idx: c.x for c in coords}
            fis = sorted(frame_to_x.keys())

            i = 0
            while i < len(fis) - 2:
                window = fis[i : i + WEAVE_WINDOW_FRAMES]
                xs     = [frame_to_x[f] for f in window]
                changes = 0
                last_dir = None
                for j in range(1, len(xs)):
                    dx = xs[j] - xs[j-1]
                    if abs(dx) < 0.3:   # ignore micro-jitter
                        continue
                    this_dir = "right" if dx > 0 else "left"
                    if last_dir and this_dir != last_dir:
                        changes += 1
                    last_dir = this_dir
                if changes >= 2:
                    events.append(PenaltyEvent(
                        type      = PenaltyType.WEAVING,
                        frame_idx = fis[i],
                        car_id    = tid,
                        details   = {"direction_changes": changes,
                                     "window_frames": WEAVE_WINDOW_FRAMES},
                        severity  = "investigation",
                    ))
                    i += WEAVE_WINDOW_FRAMES
                else:
                    i += 10
        return events

    # ── Pushing off track ─────────────────────────────────────────────────────

    def detect_pushing_off_track(self) -> list[PenaltyEvent]:
        """
        Detect when two cars are side-by-side and one subsequently goes
        off-track — suggests the other was pushed.
        """
        events = []
        tids   = list(self.car_positions.keys())
        frame_to_pos: dict[int, dict[int, tuple]] = {}
        for tid, coords in self.car_positions.items():
            for c in coords:
                frame_to_pos.setdefault(c.frame_idx, {})[tid] = (c.x, c.y)

        for fi, at in frame_to_pos.items():
            car_ids = list(at.keys())
            for i in range(len(car_ids)):
                for j in range(i+1, len(car_ids)):
                    a, b = car_ids[i], car_ids[j]
                    pa   = np.array(at[a])
                    pb   = np.array(at[b])
                    dist = float(np.linalg.norm(pa - pb))
                    if dist > PUSH_PROXIMITY_M:
                        continue
                    # Check if either car goes off-track shortly after
                    off_track_car = self._went_off_track(fi, a, b, frame_to_pos)
                    if off_track_car is not None:
                        pusher = b if off_track_car == a else a
                        events.append(PenaltyEvent(
                            type      = PenaltyType.PUSHING_OFF_TRACK,
                            frame_idx = fi,
                            car_id    = pusher,
                            details   = {"pushed_car": off_track_car, "gap_m": round(dist,1)},
                            severity  = "investigation",
                        ))
        return self._deduplicate(events, cooldown=90)

    def _went_off_track(
        self, fi: int, a: int, b: int,
        frame_to_pos: dict[int, dict[int, tuple]],
        lookahead: int = 30,
    ) -> int | None:
        """Return car_id that went off-track within lookahead frames, or None."""
        # Look for YOLO off_track detection
        for look_fi in range(fi, min(fi + lookahead, self._n_frames)):
            for det in self._yolo_by_frame.get(look_fi, []):
                if det.get("class_id") == OFF_TRACK_CLASS_ID:
                    # Which car is closest to this detection?
                    det_cx = det.get("cx", 0.5)
                    det_cy = det.get("cy", 0.5)
                    at = frame_to_pos.get(look_fi, {})
                    for cid in [a, b]:
                        if cid in at:
                            return cid
        return None

    # ── Unfair overtake ───────────────────────────────────────────────────────

    def detect_unfair_overtakes(self) -> list[PenaltyEvent]:
        """
        Overtake is flagged as potentially unfair when:
         - It occurs while a yellow flag is active, OR
         - The overtaking car is simultaneously off-track
        """
        events = []
        from analysis.race_stats import RaceStats, EventType

        # Build running order to find overtake frames
        frame_to_pos: dict[int, dict[int, tuple]] = {}
        for tid, coords in self.car_positions.items():
            for c in coords:
                frame_to_pos.setdefault(c.frame_idx, {})[tid] = (c.x, c.y)

        for fi in range(1, self._n_frames):
            yellow_active = any(
                d.get("class_id") == YELLOW_FLAG_CLASS_ID
                for d in self._yolo_by_frame.get(fi, [])
            )
            off_track_cars = {
                d.get("track_id")
                for d in self._yolo_by_frame.get(fi, [])
                if d.get("class_id") == OFF_TRACK_CLASS_ID
            }
            if not yellow_active and not off_track_cars:
                continue

            # Check for position swaps
            prev = frame_to_pos.get(fi-1, {})
            curr = frame_to_pos.get(fi,   {})
            prev_ranked = sorted(prev.items(), key=lambda kv: -kv[1][1])
            curr_ranked = sorted(curr.items(), key=lambda kv: -kv[1][1])
            prev_order  = {tid: pos for pos,(tid,_) in enumerate(prev_ranked)}
            curr_order  = {tid: pos for pos,(tid,_) in enumerate(curr_ranked)}

            common = set(prev_order) & set(curr_order)
            for a in common:
                for b in common:
                    if a >= b:
                        continue
                    if prev_order[a] > prev_order[b] and curr_order[a] < curr_order[b]:
                        reason = []
                        if yellow_active:   reason.append("yellow_flag")
                        if a in off_track_cars: reason.append("off_track")
                        if reason:
                            events.append(PenaltyEvent(
                                type      = PenaltyType.UNFAIR_OVERTAKE,
                                frame_idx = fi,
                                car_id    = a,
                                details   = {"overtaken_car": b, "reason": reason},
                                severity  = "investigation",
                            ))
        return self._deduplicate(events, cooldown=60)

    # ── Track limit violations ────────────────────────────────────────────────

    def detect_track_limit_violations(self) -> list[PenaltyEvent]:
        """
        Flags cars where YOLO detects off_track while also going fast
        (distinguishes racing off-track from pit lane).
        """
        events = []
        for fi, dets in self._yolo_by_frame.items():
            for d in dets:
                if d.get("class_id") != OFF_TRACK_CLASS_ID:
                    continue
                # Match to a tracked car by proximity (rough)
                tid  = d.get("track_id")
                spd  = 0.0
                if tid is not None:
                    s = self.car_speeds.get(tid, [])
                    spd = s[fi] if fi < len(s) else 0.0
                MIN_SPEED_KMH = 50.0
                if spd > MIN_SPEED_KMH:
                    events.append(PenaltyEvent(
                        type      = PenaltyType.TRACK_LIMIT_VIOLATION,
                        frame_idx = fi,
                        car_id    = tid or -1,
                        details   = {"speed_kmh": round(spd,1)},
                        severity  = "warning",
                    ))
        return self._deduplicate(events, cooldown=30)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(
        events: list[PenaltyEvent],
        cooldown: int = 60,
    ) -> list[PenaltyEvent]:
        """Remove duplicate events (same type + car) within cooldown frames."""
        last: dict[tuple, int] = {}
        out  = []
        for ev in sorted(events, key=lambda e: e.frame_idx):
            key = (ev.type, ev.car_id)
            if ev.frame_idx - last.get(key, -9999) >= cooldown:
                out.append(ev)
                last[key] = ev.frame_idx
        return out
