"""
main.py — F1 Race Analysis Pipeline
=====================================
Full 7-stage pipeline adapted from abdullahtarek/tennis_analysis:

  1. Video ingest & preprocessing
  2. Car detection (YOLOv8 fine-tuned)
  3. Multi-object tracking (ByteTrack)
  4a. Track keypoint detection → homography
  4b. Broadcast OCR → leaderboard
  5. Analytics engine (speed, overtakes, pit stops, penalties)
  6. Commentary generation (rule-based NLG)
  7. Visualiser (annotated video output)

Usage:
    # Full run
    python main.py --input input_videos/race.mp4 --output output_videos/output.mp4

    # With stub caching (skip re-running tracker on second pass)
    python main.py --input race.mp4 --save-stub
    python main.py --input race.mp4 --read-stub --output output.mp4

    # Skip OCR (no Tesseract installed)
    python main.py --input race.mp4 --skip-ocr

    # Custom model and circuit
    python main.py --input race.mp4 --detector models/car_detector.pt \
                   --keypoint models/keypoint_cnn.pth --circuit monaco
"""

import argparse
import time
from pathlib import Path
import os 
from trackers.car_tracker    import CarTracker, smooth_class_ids

from utils.video_utils       import read_video, save_video, get_video_fps, draw_annotations
from trackers.car_tracker    import CarTracker
from track_mapper.keypoint_detector import KeypointDetector
from track_mapper.mini_track import MiniTrack
from ocr.overlay_parser      import OverlayParser
from analysis.race_stats     import RaceStats
from analysis.penalty_detector import PenaltyDetector
from commentary.generator    import CommentaryGenerator


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="F1 Race Analysis Pipeline")
    p.add_argument("--input",    "-i", required=True,
                   help="Input video path")
    p.add_argument("--output",   "-o", default="output_videos/output.mp4",
                   help="Output annotated video path")
    p.add_argument("--detector", default="models/car_detector.pt",
                   help="YOLOv8 car detector weights")
    p.add_argument("--keypoint", default="models/keypoint_cnn.pth",
                   help="Track keypoint CNN weights (optional)")
    p.add_argument("--circuit",  default="silverstone",
                   help="Circuit name for mini-track map")
    p.add_argument("--broadcast", default="default",
                   choices=["sky_f1","f1tv","default"],
                   help="Broadcast layout for OCR region detection")
    p.add_argument("--conf",     type=float, default=0.10,
                   help="YOLO detection confidence threshold")
    p.add_argument("--device",   default="",
                   help="'' = auto, 'cpu', '0' …")
    p.add_argument("--stub-dir", default="stubs",
                   help="Directory for tracker stub cache files")
    p.add_argument("--save-stub", action="store_true",
                   help="Save tracker results to stub cache")
    p.add_argument("--read-stub", action="store_true",
                   help="Load tracker results from stub cache (skip detection)")
    p.add_argument("--skip-ocr", action="store_true",
                   help="Skip Tesseract OCR (faster, no leaderboard)")
    p.add_argument("--skip-keypoint", action="store_true",
                   help="Skip keypoint CNN (no mini-map or speed estimation)")
    p.add_argument("--show-minimap", action="store_true",   # ← ADD THIS
                   help="Show mini-map overlay (hidden by default)")
    p.add_argument("--kp-stride", type=int, default=30,
                   help="Detect keypoints every N frames (default 30)")
    p.add_argument("--ocr-stride", type=int, default=30,
                   help="Parse leaderboard every N frames (default 30)")
    p.add_argument("--audio", action="store_true",
                   help="Add spoken audio commentary to output video")
    p.add_argument("--tts-backend", default="gtts",
                   choices=["gtts", "pyttsx3", "say"],
                   help="Text-to-speech backend (default: gtts)")
    p.add_argument("--hold-frames", type=int, default=120,
                   help="How long each commentary line stays on screen in frames (default: 120 = 4s at 30fps)")
    p.add_argument("--race-direction", default="auto",
               choices=["auto", "left", "right", "up", "down"],
               help="Direction cars travel on screen (default: auto-detect). "
                    "Set manually if auto-detect is wrong: "
                    "left=cars move right-to-left, right=left-to-right, "
                    "up=bottom-to-top, down=top-to-bottom")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0   = time.time()
    args = parse_args()

    stub_path = Path(args.stub_dir) / (Path(args.input).stem + "_tracks.pkl")
    stub_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Ingest ──────────────────────────────────────────────────────────────
    print("\n[1/7] Reading video …")
    frames = read_video(args.input)
    fps    = get_video_fps(args.input)
    print(f"      {len(frames)} frames  {fps:.0f} fps")

    # ── 2+3. Detection + Tracking ──────────────────────────────────────────────
    print("[2/7] Car detection + tracking …")
    tracker    = CarTracker(
        model_path = args.detector,
        conf       = args.conf,
        device     = args.device,
    )
    car_tracks = tracker.get_car_tracks(
        frames,
        stub_path  = str(stub_path) if (args.save_stub or args.read_stub) else None,
        batch_size = 8,
    )

    # Smooth class IDs to prevent lighting-induced flickering
    car_tracks = smooth_class_ids(car_tracks, window=15)

    n_ids = len({tid for fd in car_tracks for tid in fd})
    print(f"      {n_ids} unique car IDs tracked")

    # Extract event-class detections — skip interpolated frames (conf=0.0)
    print("  Extracting event detections from tracks …")
    EVENT_CLASS_IDS = {
        12: "crash", 13: "penalty_car", 14: "pitstop",
        15: "race_start", 16: "marshal", 17: "yellow_flag", 18: "safety_car"
    }
    yolo_events = []
    for fi, frame_dict in enumerate(car_tracks):
        for tid, det in frame_dict.items():
            cls_id = det.get("class_id", 0)
            conf   = det.get("conf", 0.0)
            if cls_id in EVENT_CLASS_IDS and conf > 0.0:  # skip interpolated
                yolo_events.append({
                    "frame_idx":  fi,
                    "class_id":   cls_id,
                    "class_name": EVENT_CLASS_IDS[cls_id],
                    "track_id":   tid,
                    "conf":       conf,
                    "bbox":       det.get("bbox", []),
                })
    print(f"  Found {len(yolo_events)} event detections "
          f"({sum(1 for e in yolo_events if e['class_name']=='crash')} crash)")


    # ── 4a. Track keypoints + homography ──────────────────────────────────────
    kp_frames = {}
    if not args.skip_keypoint:
        print("[4a/7] Track keypoint detection …")
        kp_det    = KeypointDetector(
            model_path = args.keypoint if Path(args.keypoint).exists() else None,
            circuit    = args.circuit,
        )
        kp_frames = kp_det.detect_batch(frames, stride=args.kp_stride)
        n_valid   = sum(1 for r in kp_frames.values() if r.valid)
        print(f"       {n_valid}/{len(kp_frames)} valid homographies")
    else:
        print("[4a/7] Keypoint detection skipped")

    mini_track     = MiniTrack(circuit=args.circuit, keypoints_per_frame=kp_frames)
    car_positions  = mini_track.project_cars(car_tracks)

    # ── 4b. OCR ───────────────────────────────────────────────────────────────
    leaderboard_state = {}
    if not args.skip_ocr:
        print("[4b/7] Parsing broadcast leaderboard (OCR) …")
        ocr               = OverlayParser(broadcast=args.broadcast)
        leaderboard_state = ocr.parse_frames(frames, stride=args.ocr_stride)
        n_lb = sum(1 for lb in leaderboard_state.values() if lb and lb.entries)
        print(f"       {n_lb} frames with leaderboard data")
    else:
        print("[4b/7] OCR skipped")

    # ── 5. Analytics ──────────────────────────────────────────────────────────
    print("[5/7] Computing race statistics …")
    stats          = RaceStats(
        car_tracks     = car_tracks,
        car_positions  = car_positions,
        leaderboard    = leaderboard_state,
        fps            = fps,
        mini_track     = mini_track,
        detected_events  = yolo_events,  
    )
    car_speeds     = stats.compute_speeds()
    race_events    = stats.detect_events()
    speed_stats    = stats.aggregate_speed_stats()

    pen_det        = PenaltyDetector(
        car_positions  = car_positions,
        car_speeds     = car_speeds,
        mini_track     = mini_track,
        fps            = fps,
    )
    penalty_events = pen_det.detect_all()
    print(f"       {len(race_events)} race events  |  {len(penalty_events)} penalty events")


    # ── 6. Commentary ──────────────────────────────────────────────────────────
    print("[6/7] Generating commentary …")
    cgen             = CommentaryGenerator(hold_frames=args.hold_frames)
    commentary_lines = cgen.generate(
        race_events,
        penalty_events,
        leaderboard_state,
        car_tracks=car_tracks,
    )
    print(f"       {len(commentary_lines)} commentary lines")

    # ── 7. Visualise ──────────────────────────────────────────────────────────
    print("[7/7] Rendering annotated video …")
    all_events = race_events + penalty_events
    out_frames = draw_annotations(
        frames           = frames,
        car_tracks       = car_tracks,
        car_speeds       = car_speeds,
        mini_track       = mini_track,
        leaderboard_state= leaderboard_state,
        commentary_lines = commentary_lines,
        events           = all_events,
        show_minimap     = args.show_minimap,
        race_direction    = args.race_direction,
    )
    save_video(out_frames, args.output, fps=fps)


    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n── Race Summary ─────────────────────────────────────────")
    for tid, s in sorted(speed_stats.items()):
        print(f"  Car {tid:>3}  avg {s['avg_kmh']:.0f} km/h  max {s['max_kmh']:.0f} km/h")
    print(f"\n  Race events     : {len(race_events)}")
    for ev in race_events[:8]:
        print(f"    [{ev.frame_idx:>5}]  {ev}")
    if len(race_events) > 8:
        print(f"    … and {len(race_events)-8} more")
    print(f"\n  Penalty events  : {len(penalty_events)}")
    for ev in penalty_events[:5]:
        print(f"    [{ev.frame_idx:>5}]  {ev}")
    print(f"\n  Output          : {args.output}")
    print(f"  Total time      : {elapsed:.1f}s")
    print(f"────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
