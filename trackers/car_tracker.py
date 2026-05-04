"""
trackers/car_tracker.py
========================
YOLOv8 + ByteTrack multi-car tracker.

Analogous to trackers/player_tracker.py in abdullahtarek/tennis_analysis
but supports up to 20 concurrent objects (full F1 grid) and handles
the full 21-class schema.

Output format:
    car_tracks = [
        {car_id: {"bbox": [x1,y1,x2,y2], "conf": float, "class_id": int}},
        ...  # one dict per frame
    ]
"""

from __future__ import annotations
import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Classes that represent cars on track (not surface/event labels)
CAR_CLASS_IDS = {0,1,2,3,4,5,6,7,8,9,10}   # car + all team IDs

UNIFIED_CLASSES = [
    "car","RedBull","Mercedes","Ferrari","McLaren","Alpine","AstonMartin",
    "Williams","Haas","KickSauber","RacingBulls",
    "track_surface","crash","penalty_car","pitstop","race_start","marshal",
    "yellow_flag","safety_car","off_track","on_track",
]


@dataclass
class CarDetection:
    bbox:     list[float]   # [x1, y1, x2, y2] pixels
    conf:     float
    class_id: int
    track_id: int = -1
    interpolated: bool = False

    @property
    def class_name(self) -> str:
        if 0 <= self.class_id < len(UNIFIED_CLASSES):
            return UNIFIED_CLASSES[self.class_id]
        return f"cls_{self.class_id}"

    def centroid(self) -> tuple[float, float]:
        return (self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2

    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


class CarTracker:
    """
    Wraps YOLOv8 + ByteTrack to produce persistent car identities.

    Usage:
        tracker = CarTracker("models/car_detector.pt")
        tracks  = tracker.get_car_tracks(frames)          # list[dict[int, dict]]
        # or with stub caching:
        tracks  = tracker.get_car_tracks(frames, stub_path="stubs/tracks.pkl")
    """

    CONF_THRESHOLD  = 0.25
    IOU_THRESHOLD   = 0.45
    MAX_LOST_FRAMES = 30     # hold a track this many frames without detection

    def __init__(
        self,
        model_path:  str   = "models/car_detector.pt",
        conf:        float = CONF_THRESHOLD,
        iou:         float = IOU_THRESHOLD,
        car_only:    bool  = True,   # filter to car-class detections only
        device:      str   = "",
    ):
        self.conf     = conf
        self.iou      = iou
        self.car_only = car_only
        self.device   = device
        self._model   = None
        self._model_path = model_path

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("pip install ultralytics")
        self._model = YOLO(self._model_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_car_tracks(
        self,
        frames:     list[np.ndarray],
        stub_path:  str | None = None,
        batch_size: int = 8,
    ) -> list[dict[int, dict]]:
        """
        Run detection + tracking on all frames.
        If stub_path is given and exists, load from cache instead of running.

        Returns
        -------
        List[dict[int, dict]]:  one dict per frame mapping
            track_id → {"bbox": [x1,y1,x2,y2], "conf": float, "class_id": int}
        """
        # ── Stub cache ────────────────────────────────────────────────────────
        if stub_path:
            stub = Path(stub_path)
            if stub.exists():
                print(f"  [Tracker] Loading from stub: {stub}")
                with open(stub, "rb") as f:
                    return pickle.load(f)

        # ── Detection + tracking ──────────────────────────────────────────────
        self._load_model()
        print(f"  [Tracker] Running YOLOv8 + ByteTrack on {len(frames)} frames …")

        raw_results = []
        for i in range(0, len(frames), batch_size):
            batch   = frames[i : i + batch_size]
            results = self._model.track(
                batch,
                persist   = True,
                conf      = self.conf,
                iou       = self.iou,
                tracker   = "bytetrack.yaml",
                classes   = list(CAR_CLASS_IDS) if self.car_only else None,
                device    = self.device or None,
                verbose   = False,
            )
            raw_results.extend(results)
            if (i // batch_size + 1) % 10 == 0:
                print(f"    … {min(i + batch_size, len(frames))}/{len(frames)} frames")

        tracks = self._parse_results(raw_results)
        tracks = self._interpolate(tracks)

        # ── Save stub ─────────────────────────────────────────────────────────
        if stub_path:
            Path(stub_path).parent.mkdir(parents=True, exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
            print(f"  [Tracker] Saved stub → {stub_path}")

        return tracks

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _parse_results(self, results) -> list[dict[int, dict]]:
        per_frame = []
        for r in results:
            frame_dict: dict[int, dict] = {}
            if r.boxes is not None and r.boxes.id is not None:
                boxes    = r.boxes.xyxy.cpu().numpy()
                confs    = r.boxes.conf.cpu().numpy()
                ids      = r.boxes.id.cpu().numpy().astype(int)
                cls_ids  = r.boxes.cls.cpu().numpy().astype(int)
                for box, conf, tid, cid in zip(boxes, confs, ids, cls_ids):
                    frame_dict[int(tid)] = {
                        "bbox":     box.tolist(),
                        "conf":     float(conf),
                        "class_id": int(cid),
                    }
            per_frame.append(frame_dict)
        return per_frame

    def _interpolate(
        self,
        tracks:  list[dict[int, dict]],
        max_gap: int = 30,
    ) -> list[dict[int, dict]]:
        """
        Linear bbox interpolation for occluded cars.
        Mirrors ball interpolation in the tennis reference implementation.
        Interpolated frames are marked with conf=0.0.
        """
        all_ids: set[int] = set()
        for fd in tracks:
            all_ids.update(fd.keys())

        for tid in all_ids:
            visible = [
                (fi, tracks[fi][tid]["bbox"], tracks[fi][tid]["class_id"])
                for fi, fd in enumerate(tracks)
                if tid in fd
            ]
            if len(visible) < 2:
                continue

            for (f0, b0, cid), (f1, b1, _) in zip(visible, visible[1:]):
                gap = f1 - f0
                if 1 < gap <= max_gap:
                    for f in range(f0 + 1, f1):
                        t  = (f - f0) / gap
                        ib = [b0[i] + t * (b1[i] - b0[i]) for i in range(4)]
                        tracks[f][tid] = {
                            "bbox":     ib,
                            "conf":     0.0,    # marks interpolated
                            "class_id": cid,
                        }
        return tracks

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def bbox_centroid(bbox: list[float]) -> tuple[float, float]:
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    @staticmethod
    def bbox_area(bbox: list[float]) -> float:
        return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

    @staticmethod
    def filter_by_region(
        tracks:  list[dict[int, dict]],
        region:  tuple[int, int, int, int],
    ) -> list[dict[int, dict]]:
        """Remove detections whose centroid falls inside region (x1,y1,x2,y2)."""
        rx1, ry1, rx2, ry2 = region
        filtered = []
        for fd in tracks:
            new_fd = {}
            for tid, det in fd.items():
                cx, cy = CarTracker.bbox_centroid(det["bbox"])
                if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                    new_fd[tid] = det
            filtered.append(new_fd)
        return filtered

    @staticmethod
    def get_class_at_frame(tracks: list[dict], track_id: int) -> int:
        """Return the most common class_id for a track across all frames."""
        from collections import Counter
        counts = Counter(
            fd[track_id]["class_id"]
            for fd in tracks
            if track_id in fd and fd[track_id]["conf"] > 0
        )
        return counts.most_common(1)[0][0] if counts else 0
    

def smooth_class_ids(
    tracks:     list[dict[int, dict]],
    window:     int = 15,
) -> list[dict[int, dict]]:
    """
    For each track ID, replace per-frame class_id with the most common
    class_id seen in the last `window` frames. Prevents flickering
    between team labels caused by lighting/shadow changes.
    """
    from collections import Counter, defaultdict

    # Build history: {track_id: [class_id, class_id, ...]}
    history: dict[int, list[int]] = defaultdict(list)

    smoothed = []
    for fi, frame_dict in enumerate(tracks):
        new_frame = {}
        for tid, det in frame_dict.items():
            cid = det.get("class_id", 0)
            # Only smooth non-zero class IDs (skip generic "car" class 0)
            if cid > 0:
                history[tid].append(cid)
                # Keep only last `window` frames
                if len(history[tid]) > window:
                    history[tid] = history[tid][-window:]
                # Use most common class in window
                most_common = Counter(history[tid]).most_common(1)[0][0]
                new_det = dict(det)
                new_det["class_id"] = most_common
                new_frame[tid] = new_det
            else:
                history[tid].append(cid)
                new_frame[tid] = det
        smoothed.append(new_frame)

    return smoothed
