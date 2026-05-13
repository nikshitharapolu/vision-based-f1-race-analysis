# From Lights Out to Chequered Flag 🏁
### Vision-Based F1 Race Analysis Using YOLOv8 and ByteTrack

> An end-to-end computer vision pipeline that automatically detects F1 cars by team, tracks their positions, recognises race events, and generates live commentary from raw broadcast video — no human input required.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8s-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)

---

## Project Overview

Given a raw F1 broadcast video, the pipeline produces a fully annotated output video containing:

- **Team-coloured bounding boxes** with class name and estimated speed
- **Live top-3 race position scoreboard** updated every frame
- **Chat-style commentary panel** with event-triggered natural language commentary
- **Crash and yellow flag banners** with progress bar fade

All at approximately **15 FPS on consumer hardware** — no cloud, no manual input.

---

## Project Architecture
<img width="1490" height="340" alt="682_flowchart drawio (4)" src="https://github.com/user-attachments/assets/5200ca39-6704-4fa6-b983-dbdd115dee8f" />



---

## Setup

### Requirements

- Python 3.10+
- Any OS — Windows, macOS, or Linux
- GPU recommended: Apple Silicon (MPS), NVIDIA (CUDA), or CPU fallback
- ~5 GB disk space

### Installation

```bash
git clone https://github.com/your-username/vision-based-f1-race-analysis.git
cd vision-based-f1-race-analysis

pip install ultralytics opencv-python numpy pyyaml torch torchvision

# Optional — OCR leaderboard parsing
# macOS:
brew install tesseract
# Ubuntu/Debian:
sudo apt install tesseract-ocr
# Windows: download installer from https://github.com/UB-Mannheim/tesseract/wiki
pip install pytesseract

# Optional — evaluation plots
pip install matplotlib seaborn pandas
```

And update the device flag note under Quick Start:

```markdown
# Apple Silicon
--device mps

# NVIDIA GPU
--device cuda

# CPU (any system, slower)
--device cpu
```
---

## Quick Start

Place your video in `input_videos/` and run:

```bash
# First run — detects, tracks and saves cache
python main.py \
    --input input_videos/race.mp4 \
    --detector models/car_detector.pt \
    --output output_videos/result.mp4 \
    --device mps \
    --skip-keypoint \
    --skip-ocr \
    --save-stub

# Subsequent runs — uses cached tracks (much faster)
python main.py \
    --input input_videos/race.mp4 \
    --detector models/car_detector.pt \
    --output output_videos/result_v2.mp4 \
    --device mps \
    --skip-keypoint \
    --skip-ocr \
    --read-stub
```

For NVIDIA GPU replace `--device mps` with `--device cuda`.

---

## Key CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Input video path |
| `--output` | `output_videos/output.mp4` | Output video path |
| `--detector` | `models/car_detector.pt` | YOLOv8s weights |
| `--device` | auto | `mps` / `cuda` / `cpu` |
| `--conf` | `0.25` | Detection confidence threshold |
| `--race-direction` | `auto` | `left` / `right` / `up` / `down` |
| `--skip-keypoint` | off | Skip keypoint CNN |
| `--skip-ocr` | off | Skip Tesseract OCR |
| `--save-stub` | off | Save tracker cache |
| `--read-stub` | off | Load tracker cache |
| `--hold-frames` | `120` | Commentary display duration |

---

## Training

### Build the dataset

```bash
python roboflow_dataset_builder.py
```

### Verify

```bash
python training/train_detector.py --verify --data roboflow_dataset/data.yaml
```

### Train (recommended on Google Colab T4)

```python
from ultralytics import YOLO
model = YOLO("yolov8s.pt")
model.train(
    data   = "/content/roboflow_dataset/data.yaml",
    epochs = 100,
    batch  = 16,
    imgsz  = 640,
    device = "cuda",
    project= "/content/drive/MyDrive/runs",
    name   = "f1_detector_v3",
)
```

### Evaluate

```bash
python training/train_detector.py \
    --eval-only \
    --weights models/car_detector.pt \
    --data roboflow_dataset/data.yaml \
    --eval-conf 0.10 \
    --device mps
```

---

## Dataset

9 Roboflow Universe sources merged via `LABEL_REMAP` into a unified 21-class schema:

| Split | Images |
|---|---|
| Train | 11,058 |
| Val | 2,370 |
| Test | 2,408 |

**21 Classes:** 10 F1 team identities (RedBull, Mercedes, Ferrari, McLaren, Alpine, AstonMartin, Williams, Haas, KickSauber, RacingBulls) + 11 event classes (car, track_surface, crash, penalty_car, pitstop, race_start, marshal, yellow_flag, safety_car, off_track, on_track)

---

## Key Technical Finding

**Class imbalance is a calibration problem, not a capacity problem.**

At the standard confidence threshold of 0.25, crash detection returned F1 = 0.000 despite the model correctly detecting crashes at confidence scores of 0.12–0.18. Lowering the threshold to 0.10 for rare event classes recovered crash F1 to 0.797 — same model, same weights, no retraining. This finding generalises to any sports video dataset where incidents are structurally rare.

---

## Limitations

- `on_track` / `off_track` detection weak (F1 = 0.315) due to limited annotated data
- Rule-based commentary cannot reference specific corners or adapt to race context
- Top-3 scoreboard uses screen-space heuristics — fails during replay sequences
- Validated on Miami GP footage only; cross-circuit generalisation untested

---

## Future Work

- **Vision-language API** (Claude Vision / Gemini) for frame-conditioned commentary
- **Keypoint CNN** for homography-based speed estimation in real km/h
- **Named track landmarks** for corner-specific commentary
- **Multi-circuit generalisation** across the full 24-venue F1 calendar

---

## Results

| Metric | Value |
|---|---|
| mAP @ IoU=0.50 | **0.8386**|
| F1 Score (macro) | **0.8521** |
| Precision | 0.8771 |
| Recall | 0.8285 |
| All 10 F1 teams | **F1 > 0.93** |
| Pitstop | **F1 = 1.000** |
| Crash (conf=0.10) | F1 = 0.797 |
| Speed (M4 MPS) | ~15 FPS |

---

## Output video

https://www.dropbox.com/scl/fo/x71idgrj137c665m3c3g6/ACdiEJdDxFWbHop97bkO0Pg?rlkey=fytodes2nm2145owsg5qlu766&st=3eggegkg&dl=0

---

## Project Material

Project Poster:
[Poster_682_final.pdf](https://github.com/user-attachments/files/27688290/Poster_682_final.pdf)




---

## Authors

**Nikshitha Rapolu** , **Shreya Ramarao**  
Manning College of Information & Computer Sciences, UMass Amherst

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [Roboflow Universe](https://universe.roboflow.com)
- [abdullahtarek/tennis_analysis](https://github.com/abdullahtarek/tennis_analysis) — pipeline architecture inspiration
