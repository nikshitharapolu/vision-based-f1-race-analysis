"""
analysis/race_stats.py
=======================
Speed estimation, overtake detection, pit stop detection, close battle
alerts, and crash/flag event tagging.

Analogous to analysis/player_stats.py in abdullahtarek/tennis_analysis,
extended with race-specific event detection.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ── Constants ─────────────────────────────────────────────────────────────────

SPEED_SMOOTH_WIN        = 5      # rolling-mean window (frames)
MIN_OVERTAKE_SPEED_KMH  = 50.0
PIT_ENTRY_SPEED_KMH     = 80.0
BATTLE_GAP_M            = 30.0   # metres — "close battle" threshold
OVERTAKE_COOLDOWN       = 30     # min frames between same pair overtakes
BATTLE_COOLDOWN         = 90     # frames between battle alerts (same pair)


# ── Event types ───────────────────────────────────────────────────────────────

class EventType(Enum):
    OVERTAKE      = auto()
    PIT_ENTRY     = auto()
    PIT_EXIT      = auto()
    CLOSE_BATTLE  = auto()
    CRASH         = auto()
    YELLOW_FLAG   = auto()
    RACE_START    = auto()


@dataclass
class RaceEvent:
    type:      EventType
    frame_idx: int
    car_ids:   list[int]
    details:   dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        cars = ", ".join(f"Car {c}" for c in self.car_ids)
        return f"{self.type.name:<14} [{cars}]  {self.details}"


# ── Main class ────────────────────────────────────────────────────────────────

class RaceStats:
    """
    Consumes car tracks + projected world positions.
    Emits events and per-car statistics.
    """

    def __init__(
        self,
        car_tracks:    list[dict[int, dict]],
        car_positions: dict[int, list],      # tid → list[TrackCoord]
        leaderboard:   dict[int, Any],       # frame_idx → FrameLeaderboard
        fps:           float = 30.0,
        mini_track=    None,
        detected_events: list[dict] | None = None,  # from YOLO (crash, flag…)
    ):
        self.car_tracks    = car_tracks
        self.car_positions = car_positions
        self.leaderboard   = leaderboard
        self.fps           = fps
        self.mini_track    = mini_track
        self.yolo_events   = detected_events or []
        self._n_frames     = len(car_tracks)

    # ── Speed ─────────────────────────────────────────────────────────────────

    def compute_speeds(self) -> dict[int, list[float]]:
        """
        Compute instantaneous speed in km/h for every car at every frame.
        Uses pixel displacement projected through the homography.
        """
        speeds: dict[int, list[float]] = {}
        for tid, coords in self.car_positions.items():
            frame_to_pos = {c.frame_idx: (c.x, c.y) for c in coords}
            raw = [0.0] * self._n_frames
            for fi in range(1, self._n_frames):
                if fi in frame_to_pos and (fi-1) in frame_to_pos:
                    p0  = np.array(frame_to_pos[fi-1])
                    p1  = np.array(frame_to_pos[fi])
                    dist_m   = float(np.linalg.norm(p1 - p0))
                    speed_ms = dist_m * self.fps
                    raw[fi]  = min(speed_ms * 3.6, 400.0)
            speeds[tid] = self._smooth(raw, SPEED_SMOOTH_WIN)
        return speeds

    def aggregate_speed_stats(self) -> dict[int, dict]:
        all_speeds = self.compute_speeds()
        stats = {}
        for tid, spd in all_speeds.items():
            nz = [v for v in spd if v > 1.0]
            if nz:
                stats[tid] = {
                    "avg_kmh": float(np.mean(nz)),
                    "max_kmh": float(np.max(nz)),
                    "min_kmh": float(np.min(nz)),
                }
        return stats

    # ── Running order ─────────────────────────────────────────────────────────

    def compute_running_order(self) -> list[dict[int, int]]:
        """
        Rank cars by track Y-position each frame.
        Returns [{track_id: rank}, ...].
        """
        frame_to_coords: dict[int, dict[int, tuple]] = {}
        for tid, coords in self.car_positions.items():
            for c in coords:
                frame_to_coords.setdefault(c.frame_idx, {})[tid] = (c.x, c.y)

        orders = []
        for fi in range(self._n_frames):
            at = frame_to_coords.get(fi, {})
            ranked = sorted(at.items(), key=lambda kv: -kv[1][1])
            orders.append({tid: pos+1 for pos, (tid,_) in enumerate(ranked)})
        return orders

    # ── Event detection ───────────────────────────────────────────────────────

    def detect_events(self) -> list[RaceEvent]:
        speeds = self.compute_speeds()
        orders = self.compute_running_order()
        events: list[RaceEvent] = []
        events += self._detect_overtakes(orders, speeds)
        events += self._detect_pit_events(speeds)
        events += self._detect_close_battles()
        events += self._detect_yolo_events()
        events.sort(key=lambda e: e.frame_idx)
        return events

    def _detect_overtakes(
        self, orders: list[dict], speeds: dict
    ) -> list[RaceEvent]:
        events = []
        last: dict[frozenset, int] = {}
        for fi in range(1, self._n_frames):
            prev, curr = orders[fi-1], orders[fi]
            common = set(prev) & set(curr)
            for a in common:
                for b in common:
                    if a >= b:
                        continue
                    if prev[a] > prev[b] and curr[a] < curr[b]:
                        pair = frozenset([a,b])
                        if fi - last.get(pair, -9999) < OVERTAKE_COOLDOWN:
                            continue
                        spd = speeds.get(a, [0.0]*self._n_frames)
                        if fi < len(spd) and spd[fi] < MIN_OVERTAKE_SPEED_KMH:
                            continue
                        last[pair] = fi
                        events.append(RaceEvent(
                            type      = EventType.OVERTAKE,
                            frame_idx = fi,
                            car_ids   = [a, b],
                            details   = {
                                "overtaker": a, "overtaken": b,
                                "new_pos_a": curr[a], "new_pos_b": curr[b],
                            },
                        ))
        return events

    def _detect_pit_events(self, speeds: dict) -> list[RaceEvent]:
        events = []
        if self.mini_track is None:
            return events
        for tid, coords in self.car_positions.items():
            was_in_pit = False
            spd = speeds.get(tid, [0.0]*self._n_frames)
            for c in coords:
                in_pit  = self.mini_track.is_in_pit_lane(c)
                fi_spd  = spd[c.frame_idx] if c.frame_idx < len(spd) else 0.0
                if in_pit and not was_in_pit and fi_spd < PIT_ENTRY_SPEED_KMH:
                    events.append(RaceEvent(
                        type=EventType.PIT_ENTRY, frame_idx=c.frame_idx,
                        car_ids=[tid], details={"speed_kmh": round(fi_spd,1)},
                    ))
                elif not in_pit and was_in_pit:
                    events.append(RaceEvent(
                        type=EventType.PIT_EXIT, frame_idx=c.frame_idx,
                        car_ids=[tid],
                    ))
                was_in_pit = in_pit
        return events

    def _detect_close_battles(self) -> list[RaceEvent]:
        events = []
        last: dict[frozenset, int] = {}
        frame_to_coords: dict[int, dict[int, tuple]] = {}
        for tid, coords in self.car_positions.items():
            for c in coords:
                frame_to_coords.setdefault(c.frame_idx, {})[tid] = (c.x, c.y)

        for fi, at in frame_to_coords.items():
            tids = list(at.keys())
            for i in range(len(tids)):
                for j in range(i+1, len(tids)):
                    a, b = tids[i], tids[j]
                    dist = float(np.linalg.norm(
                        np.array(at[a]) - np.array(at[b])
                    ))
                    if dist < BATTLE_GAP_M:
                        pair = frozenset([a,b])
                        if fi - last.get(pair, -9999) > BATTLE_COOLDOWN:
                            last[pair] = fi
                            events.append(RaceEvent(
                                type=EventType.CLOSE_BATTLE, frame_idx=fi,
                                car_ids=[a,b],
                                details={"gap_m": round(dist,1)},
                            ))
        return events

    def _detect_yolo_events(self) -> list[RaceEvent]:
        """
        Convert YOLO event detections to RaceEvents.
        Requires detection to appear in multiple consecutive frames
        before firing — eliminates single-frame false positives and
        early triggering from motion blur.
        """
        events = []

        CLASS_TO_EVENT = {
            12: EventType.CRASH,
            15: EventType.RACE_START,
            17: EventType.YELLOW_FLAG,
            "crash":       EventType.CRASH,
            "yellow_flag": EventType.YELLOW_FLAG,
            "race_start":  EventType.RACE_START,
        }

        # Minimum consecutive frames needed to confirm event
        CONFIRM_FRAMES = {
            EventType.CRASH:       5,   # crash must appear in 5 consecutive frames
            EventType.YELLOW_FLAG: 4,
            EventType.RACE_START:  3,
        }

        COOLDOWN = {
            EventType.CRASH:       90,
            EventType.YELLOW_FLAG: 120,
            EventType.RACE_START:  300,
        }

        # Group detections by frame
        by_frame: dict[int, list[dict]] = {}
        for det in self.yolo_events:
            fi = det.get("frame_idx", 0)
            by_frame.setdefault(fi, []).append(det)

        last_seen: dict[EventType, int] = {}
        # Track consecutive detection counts per event type
        streak: dict[EventType, int] = {}
        streak_start: dict[EventType, int] = {}

        for fi in sorted(by_frame.keys()):
            dets = by_frame[fi]
            seen_types = set()

            for det in dets:
                cls_id   = det.get("class_id", -1)
                cls_name = det.get("class_name", "")
                conf     = det.get("conf", 0.0)

                ev_type = CLASS_TO_EVENT.get(cls_id) or CLASS_TO_EVENT.get(cls_name)
                if ev_type is None:
                    continue

                seen_types.add(ev_type)

                # Build consecutive streak
                if ev_type not in streak:
                    streak[ev_type]       = 1
                    streak_start[ev_type] = fi
                else:
                    # Check if this is consecutive (within 3 frames gap)
                    last_fi = streak_start.get(ev_type, 0) + streak.get(ev_type, 0)
                    if fi <= last_fi + 3:
                        streak[ev_type] += 1
                    else:
                        # Reset streak
                        streak[ev_type]       = 1
                        streak_start[ev_type] = fi

                confirm_needed = CONFIRM_FRAMES.get(ev_type, 3)
                cooldown       = COOLDOWN.get(ev_type, 90)

                # Fire event only when streak reaches confirmation threshold
                if (streak[ev_type] >= confirm_needed and
                        fi - last_seen.get(ev_type, -9999) >= cooldown):

                    # Use the START of the streak as the event frame
                    # so the banner appears when crash first became visible
                    event_fi = streak_start[ev_type]

                    last_seen[ev_type]  = fi
                    streak[ev_type]     = 0   # reset after firing

                    events.append(RaceEvent(
                        type      = ev_type,
                        frame_idx = event_fi,
                        car_ids   = [det.get("track_id", -1)],
                        details   = {
                            "conf":  round(conf, 3),
                            "class": cls_name or str(cls_id),
                        },
                    ))

            # Reset streaks for event types not seen this frame
            for ev_type in list(streak.keys()):
                if ev_type not in seen_types:
                    # Allow small gaps (3 frames) before resetting streak
                    last_fi = streak_start.get(ev_type, 0) + streak.get(ev_type, 0)
                    if fi > last_fi + 3:
                        streak[ev_type] = 0

        print(f"  YOLO events: {len(events)} "
            f"({sum(1 for e in events if e.type == EventType.CRASH)} crash, "
            f"{sum(1 for e in events if e.type == EventType.YELLOW_FLAG)} yellow)")
        return events

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _smooth(values: list[float], window: int) -> list[float]:
        result = list(values)
        half   = window // 2
        for i in range(half, len(values) - half):
            result[i] = float(np.mean(values[i-half : i+half+1]))
        return result
