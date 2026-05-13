"""
Complete training pipeline for the F1 Race Analysis project.


  1. Dataset verification   — checks images, labels, class counts before training
  2. YOLOv8 fine-tuning     — trains the car/event detector on merged dataset
  3. Evaluation             — mAP@0.5, per-class AP, confusion matrix
  4. Keypoint CNN training  — trains the track landmark regression model  --Future scope

  
    # Step 1 — verify dataset first 
    python training/train_detector.py --verify --data roboflow_dataset/data.yaml

    # Step 2 — train the car detector
    python training/train_detector.py --data roboflow_dataset/data.yaml

    # Step 3 — evaluate an existing model
    python training/train_detector.py --eval-only --weights models/car_detector.pt \
                                      --data roboflow_dataset/data.yaml

    # Train keypoint CNN (after annotating landmarks)
    python training/train_detector.py --train-keypoint \
                                      --keypoint-annotations dataset/keypoints/

    # Quick smoke-test (5 epochs, sanity check)
    python training/train_detector.py --data roboflow_dataset/data.yaml --smoke

    # Resume interrupted training
    python training/train_detector.py --data roboflow_dataset/data.yaml --resume

"""

import argparse
import shutil
import sys
import time
from collections import Counter
from pathlib import Path


UNIFIED_CLASSES = [
    "car",           # 0
    "RedBull",       # 1
    "Mercedes",      # 2
    "Ferrari",       # 3
    "McLaren",       # 4
    "Alpine",        # 5
    "AstonMartin",   # 6
    "Williams",      # 7
    "Haas",          # 8
    "KickSauber",    # 9
    "RacingBulls",   # 10
    "track_surface", # 11
    "crash",         # 12
    "penalty_car",   # 13
    "pitstop",       # 14
    "race_start",    # 15
    "marshal",       # 16
    "yellow_flag",   # 17
    "safety_car",    # 18
    "off_track",     # 19
    "on_track",      # 20
]

# YOLOv8 training hyperparameters 

TRAIN_CFG = dict(
    epochs        = 100,
    batch         = 16,
    imgsz         = 640,
    lr0           = 0.01,
    lrf           = 0.01,
    momentum      = 0.937,
    weight_decay  = 0.0005,
    warmup_epochs = 3,
    patience      = 30,          # early stopping — stops if no improvement
    mosaic        = 1.0,         # 4-image mosaic (YOLOv8 native)
    mixup         = 0.1,
    copy_paste    = 0.1,
    flipud        = 0.0,        
    fliplr        = 0.5,         
    degrees       = 2.0,        
    translate     = 0.1,
    scale         = 0.5,
    hsv_h         = 0.030,
    hsv_s         = 0.90,
    hsv_v         = 0.60,
    
    save          = True,
    plots         = True,        
    verbose       = True,
    save_period   = 10,         
    auto_augment = "randaugment",   
    erasing      = 0.40,            
    
)


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def verify_dataset(data_yaml: str) -> bool:
    """
    Returns True if dataset looks healthy, False if critical issues found.
    """
    try:
        import yaml
    except ImportError:
        print("  [WARN] pyyaml not installed — skipping full verification")
        return True

    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"  [ERROR] data.yaml not found: {data_path}")
        return False

    cfg      = yaml.safe_load(data_path.read_text())
    root     = Path(cfg.get("path", data_path.parent))
    nc       = cfg.get("nc", 0)
    names    = cfg.get("names", {})

    print(f"\n── Dataset Verification ─────────────────────────────────")
    print(f"   Path    : {root}")
    print(f"   Classes : {nc}")
    print(f"   Names   : {list(names.values()) if isinstance(names, dict) else names}")

    if nc != len(UNIFIED_CLASSES):
        print(f"  [WARN] data.yaml has nc={nc} but UNIFIED_CLASSES has {len(UNIFIED_CLASSES)}")

    issues   = 0
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for split in ("train", "val", "test"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"

        if not img_dir.exists():
            print(f"  [WARN] Missing split directory: {img_dir}")
            continue

        images = [f for f in img_dir.iterdir() if f.suffix.lower() in img_exts]
        labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        # Count boxes per class
        class_counts: Counter = Counter()
        empty_labels = 0
        bad_labels   = 0

        for lbl_path in labels:
            lines = lbl_path.read_text().strip().splitlines()
            if not lines:
                empty_labels += 1
                continue
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    bad_labels += 1
                    continue
                try:
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]
                    if not all(0.0 <= v <= 1.0 for v in coords):
                        bad_labels += 1
                    elif cls_id >= nc:
                        bad_labels += 1
                        issues += 1
                    else:
                        class_counts[cls_id] += 1
                except ValueError:
                    bad_labels += 1

        print(f"\n   {split.upper()} split:")
        print(f"     Images      : {len(images)}")
        print(f"     Labels      : {len(labels)}")
        print(f"     Empty labels: {empty_labels}  (hard negatives — OK)")
        if bad_labels:
            print(f"     Bad lines   : {bad_labels}  [WARN]")
            issues += bad_labels
        img_stems = {f.stem for f in images}
        lbl_stems = {f.stem for f in labels}
        missing   = img_stems - lbl_stems
        if missing:
            print(f"     No label    : {len(missing)} images have no .txt file")

        total_boxes = sum(class_counts.values())
        if total_boxes > 0:
            print(f"     Total boxes : {total_boxes}")
            print(f"     Class distribution:")
            for cls_id, count in sorted(class_counts.items()):
                name = UNIFIED_CLASSES[cls_id] if cls_id < len(UNIFIED_CLASSES) else f"cls_{cls_id}"
                pct  = count / total_boxes * 100
                bar  = "█" * max(1, int(pct / 3))
                print(f"       {cls_id:>2}  {name:<18} {count:>6}  {pct:5.1f}%  {bar}")

    print(f"\n   Verification {'PASSED' if issues == 0 else f'ISSUES: {issues}'}")
    print(f"──────────────────────────────────────────────────────────\n")
    return issues == 0


# ══════════════════════════════════════════════════════════════════════════════
#  YOLOV8 TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(args) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics")
    cfg = dict(TRAIN_CFG)
    cfg["epochs"] = 5 if args.smoke else args.epochs
    cfg["batch"]  = args.batch
    cfg["imgsz"]  = args.imgsz
    if args.smoke:
        cfg.update(mosaic=0.0, mixup=0.0, copy_paste=0.0, patience=5)

    print(f"\n── F1 Car Detector — YOLOv8 Fine-tuning ─────────────────")
    print(f"   Base model : {args.model}")
    print(f"   Dataset    : {args.data}")
    print(f"   Epochs     : {cfg['epochs']}")
    print(f"   Batch      : {cfg['batch']}")
    print(f"   Image size : {cfg['imgsz']}")
    print(f"   Classes    : {len(UNIFIED_CLASSES)}")
    print(f"   Device     : {args.device or 'auto (GPU if available)'}")
    print(f"   Output     : {args.project}/{args.name}")
    if args.smoke:
        print(f"   Mode       : SMOKE TEST (5 epochs, no augmentation)")
    print(f"──────────────────────────────────────────────────────────\n")

    # Load pretrained model — downloads COCO weights automatically on first run
    model = YOLO(args.model)

    t0 = time.time()
    model.train(
        data    = args.data,
        project = args.project,
        name    = args.name,
        resume  = args.resume,
        device  = args.device or None,
        **cfg,
    )
    elapsed = time.time() - t0

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if best.exists():
        Path("models").mkdir(exist_ok=True)
        dst = Path("models/car_detector.pt")
        shutil.copy2(best, dst)
        print(f"\n  Best weights saved → {dst}")
        print(f"  Training time      : {elapsed/60:.1f} min")
        print(f"\n  To run the pipeline:")
        print(f"    python main.py --input your_video.mp4 --detector {dst}")
    else:
        print(f"  [WARN] best.pt not found at {best}")


# ══════════════════════════════════════════════════════════════════════════════
#  COMPREHENSIVE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(weights: str, data: str, imgsz: int = 640, split: str = "test",
             save_report: bool = True, conf: float = 0.25, iou: float = 0.45) -> None:
    """
    Full evaluation suite:
      - Overall: mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1
      - Per-class: Precision, Recall, F1, AP@0.5, AP@0.5:0.95
      - Confusion matrix 
      - Accuracy estimate (TP / total predictions)
      - Summary: best/worst classes, macro/weighted averages
      - Saves full report to runs/detect/eval_report.txt
    """
    try:
        from ultralytics import YOLO
        import numpy as np
    except ImportError:
        raise ImportError("pip install ultralytics numpy")

    SEP   = "─" * 70
    SEP2  = "═" * 70

    print(f"\n{SEP2}")
    print(f"  EVALUATION REPORT")
    print(f"  Model  : {weights}")
    print(f"  Split  : {split}")
    print(f"  Conf   : {conf}   IoU: {iou}")
    print(f"{SEP2}\n")

    model   = YOLO(weights)
    metrics = model.val(
        data    = data,
        split   = split,
        imgsz   = imgsz,
        conf    = conf,
        iou     = iou,
        verbose = False,
        plots   = True,    
    )

    box = metrics.box

    # Overall metrics 
    P       = float(box.mp)                  
    R       = float(box.mr)                  
    F1      = 2 * P * R / (P + R + 1e-9)  
    mAP50   = float(box.map50)
    mAP5095 = float(box.map)

    TP_total = 0.0
    FP_total = 0.0
    FN_total = 0.0
    if hasattr(box, "p") and box.p is not None and len(box.p) > 0:
        per_p = np.array(box.p,  dtype=float)
        per_r = np.array(box.r,  dtype=float)
        per_ap50 = np.array(box.ap50, dtype=float) if hasattr(box, "ap50") else per_p
        TP_total  = per_r.sum()
        FP_total  = (1 - per_p + 1e-9).sum()
        FN_total  = (1 - per_r + 1e-9).sum()

    accuracy = TP_total / (TP_total + FP_total + 1e-9)

    print(f"  {'OVERALL METRICS':}")
    print(f"  {SEP}")
    print(f"  {'Metric':<28} {'Value':>10}  {'Interpretation'}")
    print(f"  {SEP}")
    print(f"  {'mAP @ IoU=0.50':<28} {mAP50:>10.4f}  {'Excellent >0.80 | Good >0.60 | Weak <0.40'}")
    print(f"  {'mAP @ IoU=0.50:0.95':<28} {mAP5095:>10.4f}  {'Strict — COCO standard'}")
    print(f"  {'Precision (mean)':<28} {P:>10.4f}  {'Of boxes predicted, % correct'}")
    print(f"  {'Recall (mean)':<28} {R:>10.4f}  {'Of real objects, % found'}")
    print(f"  {'F1 Score (macro)':<28} {F1:>10.4f}  {'Harmonic mean of P & R'}")
    print(f"  {'Accuracy (approx)':<28} {accuracy:>10.4f}  {'TP / (TP+FP) at conf={conf}'}")
    print()

    if hasattr(box, "ap_class_index") and box.ap_class_index is not None and len(box.ap_class_index) > 0:

        cls_indices = list(box.ap_class_index)
        per_p_arr   = np.array(box.p,    dtype=float) if hasattr(box, "p")    else np.zeros(len(cls_indices))
        per_r_arr   = np.array(box.r,    dtype=float) if hasattr(box, "r")    else np.zeros(len(cls_indices))
        per_ap50_arr= np.array(box.ap50, dtype=float) if hasattr(box, "ap50") else np.zeros(len(cls_indices))
        per_ap_arr  = np.array(box.ap,   dtype=float) if hasattr(box, "ap")   else np.zeros(len(cls_indices))

        # Align arrays to cls_indices length
        def safe(arr, i): return float(arr[i]) if i < len(arr) else 0.0

        print(f"  PER-CLASS METRICS")
        print(f"  {SEP}")
        header = f"  {'Class':<18} {'Prec':>7} {'Recall':>7} {'F1':>7} {'AP@50':>7} {'AP50:95':>8}  Bar"
        print(header)
        print(f"  {SEP}")

        class_rows = []
        for i, cls_idx in enumerate(cls_indices):
            cls_name = UNIFIED_CLASSES[cls_idx] if cls_idx < len(UNIFIED_CLASSES) else f"cls_{cls_idx}"
            p_i   = safe(per_p_arr,    i)
            r_i   = safe(per_r_arr,    i)
            f1_i  = 2 * p_i * r_i / (p_i + r_i + 1e-9)
            ap50  = safe(per_ap50_arr, i)
            ap    = safe(per_ap_arr,   i)
            bar   = "█" * max(1, int(ap50 * 25))
            class_rows.append((cls_name, p_i, r_i, f1_i, ap50, ap))
            print(f"  {cls_name:<18} {p_i:>7.4f} {r_i:>7.4f} {f1_i:>7.4f} {ap50:>7.4f} {ap:>8.4f}  {bar}")

        n = len(class_rows)
        if n > 0:
            macro_p  = sum(r[1] for r in class_rows) / n
            macro_r  = sum(r[2] for r in class_rows) / n
            macro_f1 = sum(r[3] for r in class_rows) / n
            macro_ap = sum(r[4] for r in class_rows) / n
            print(f"  {SEP}")
            print(f"  {'MACRO AVERAGE':<18} {macro_p:>7.4f} {macro_r:>7.4f} {macro_f1:>7.4f} {macro_ap:>7.4f}")

        sorted_by_f1 = sorted(class_rows, key=lambda x: x[3], reverse=True)
        print(f"\n  TOP 5 CLASSES BY F1:")
        for row in sorted_by_f1[:5]:
            print(f"    {row[0]:<18}  F1={row[3]:.4f}  AP@50={row[4]:.4f}")
        print(f"\n  BOTTOM 5 CLASSES BY F1 (need more data):")
        for row in sorted_by_f1[-5:]:
            print(f"    {row[0]:<18}  F1={row[3]:.4f}  AP@50={row[4]:.4f}")

    print(f"\n  CONFUSION MATRIX")
    print(f"  Saved as image → runs/detect/val/confusion_matrix.png")
    print(f"  (YOLOv8 --plots=True generates this automatically)")

    print(f"\n  THRESHOLD GUIDANCE")
    print(f"  {SEP}")
    print(f"  Current conf={conf} gives P={P:.4f}, R={R:.4f}, F1={F1:.4f}")
    if P > 0.85 and R < 0.60:
        print(f"  → Precision-heavy: lower --conf to catch more objects (try 0.15)")
    elif R > 0.85 and P < 0.60:
        print(f"  → Recall-heavy: raise --conf to reduce false positives (try 0.40)")
    elif F1 > 0.70:
        print(f"  → Good balance. Current threshold is appropriate.")
    else:
        print(f"  → Low F1: consider more training data for underperforming classes.")

    grade = ("A — Production ready" if mAP50 >= 0.80 else
             "B — Good, minor improvements needed" if mAP50 >= 0.65 else
             "C — Acceptable, add data for weak classes" if mAP50 >= 0.50 else
             "D — Needs significantly more training data")
    print(f"\n  OVERALL GRADE:  {grade}")
    print(f"\n  Plots saved to: runs/detect/val/")
    print(f"    confusion_matrix.png  PR_curve.png  F1_curve.png  R_curve.png")
    print(f"{SEP2}\n")

    # Save text report 
    if save_report:
        import datetime
        report_dir = Path("runs/detect/val")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "eval_report.txt"

        lines = [
            "F1 Race Analysis — Evaluation Report",
            f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model     : {weights}",
            f"Dataset   : {data}  split={split}",
            f"Conf      : {conf}   IoU: {iou}",
            "",
            "OVERALL METRICS",
            f"  mAP@0.50       : {mAP50:.4f}",
            f"  mAP@0.50:0.95  : {mAP5095:.4f}",
            f"  Precision      : {P:.4f}",
            f"  Recall         : {R:.4f}",
            f"  F1 Score       : {F1:.4f}",
            f"  Accuracy (est) : {accuracy:.4f}",
            f"  Grade          : {grade}",
            "",
            "PER-CLASS METRICS",
            f"  {'Class':<18} {'Prec':>7} {'Recall':>7} {'F1':>7} {'AP@50':>7} {'AP50:95':>8}",
        ]
        if hasattr(box, "ap_class_index") and box.ap_class_index is not None:
            for row in class_rows:
                lines.append(
                    f"  {row[0]:<18} {row[1]:>7.4f} {row[2]:>7.4f} "
                    f"{row[3]:>7.4f} {row[4]:>7.4f} {row[5]:>8.4f}"
                )

        report_path.write_text("\n".join(lines))
        print(f"  Text report → {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  KEYPOINT CNN TRAINING 
# ══════════════════════════════════════════════════════════════════════════════

def train_keypoint(annotations_dir: str, circuit: str = "default",
                   epochs: int = 50, lr: float = 1e-4,
                   save_path: str = "models/keypoint_cnn.pth") -> None:
    """
    Fine-tune a ResNet-50 to regress track landmark coordinates.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import torchvision.transforms as T
        from torchvision import models
        from torch.utils.data import DataLoader, Dataset
        from PIL import Image as PILImage
        import json, numpy as np
    except ImportError:
        raise ImportError("pip install torch torchvision Pillow numpy")

    CIRCUIT_KP = {"silverstone":8,"monaco":10,"monza":8,"spa":8,"default":8}
    n_kp       = CIRCUIT_KP.get(circuit, 8)
    ann_dir    = Path(annotations_dir)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load annotation records
    records = []
    for jf in sorted(ann_dir.glob("*.json")):
        d        = json.loads(jf.read_text())
        img_path = ann_dir.parent / "images" / d["image"]
        if not img_path.exists():
            print(f"  [WARN] Image not found: {img_path}")
            continue
        kps = np.array(d["keypoints"], dtype=np.float32)
        if len(kps) != n_kp:
            print(f"  [WARN] {jf.name}: expected {n_kp} keypoints, got {len(kps)}")
            continue
        records.append((str(img_path), kps))

    if not records:
        print(f"  [ERROR] No valid annotation JSONs found in {ann_dir}")
        print(f"  Each JSON must have keys: 'image' (filename) and 'keypoints' (list of [x,y])")
        return

    print(f"\n── Keypoint CNN Training ────────────────────────────────")
    print(f"   Circuit    : {circuit}  ({n_kp} landmarks)")
    print(f"   Annotations: {len(records)}")
    print(f"   Epochs     : {epochs}")
    print(f"   Device     : {device}")
    print(f"──────────────────────────────────────────────────────────\n")

    class KPDataset(Dataset):
        def __init__(self, records):
            self.records = records
            self.tf = T.Compose([
                T.Resize((224, 224)),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
        def __len__(self): return len(self.records)
        def __getitem__(self, i):
            img_path, kps = self.records[i]
            img   = PILImage.open(img_path).convert("RGB")
            w, h  = img.size
            kps_n = kps / np.array([w, h], dtype=np.float32)  # normalise 
            return self.tf(img), torch.tensor(kps_n.flatten(), dtype=torch.float32)

    # Split 80/20
    split_n = int(len(records) * 0.8)
    train_r, val_r = records[:split_n], records[split_n:]
    train_loader   = DataLoader(KPDataset(train_r), batch_size=16, shuffle=True)
    val_loader     = DataLoader(KPDataset(val_r),   batch_size=16, shuffle=False)

    # ResNet-50 with regression head
    m     = models.resnet50(weights="IMAGENET1K_V2")
    m.fc  = nn.Linear(m.fc.in_features, n_kp * 2)
    m.to(device)

    opt     = optim.Adam(m.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val_loss = float("inf")

    for ep in range(epochs):
        # Train
        m.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            loss = loss_fn(m(imgs), targets)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item()

        # Validate
        m.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                val_loss += loss_fn(m(imgs), targets).item()

        train_loss /= max(len(train_loader), 1)
        val_loss   /= max(len(val_loader), 1)

        print(f"  Epoch {ep+1:>3}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

        # Save best epoch
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(m.state_dict(), save_path)

    print(f"\n  Best val loss : {best_val_loss:.6f}")
    print(f"  Saved → {save_path}")
    print(f"\n  To use in the pipeline:")
    print(f"    python main.py --input video.mp4 --keypoint {save_path} --circuit {circuit}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="F1 Race Analysis — Model Training",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Always verify first
  python training/train_detector.py --verify --data roboflow_dataset/data.yaml

  # Train
  python training/train_detector.py --data roboflow_dataset/data.yaml

  # Evaluate
  python training/train_detector.py --eval-only --weights models/car_detector.pt

  # Train keypoint CNN
  python training/train_detector.py --train-keypoint --keypoint-annotations dataset/keypoints/

  # Resume interrupted training
  python training/train_detector.py --data roboflow_dataset/data.yaml --resume
        """
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--verify",          action="store_true",
                      help="Verify dataset integrity only (recommended before training)")
    mode.add_argument("--eval-only",       action="store_true",
                      help="Evaluate existing weights, skip training")
    mode.add_argument("--train-keypoint",  action="store_true",
                      help="Train the track keypoint CNN instead of the car detector")

    # Detector training
    p.add_argument("--data",    default="roboflow_dataset/data.yaml",
                   help="Path to data.yaml  (default: roboflow_dataset/data.yaml)")
    p.add_argument("--model",   default="yolov8s.pt",
                   choices=["yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt"],
                   help="Base YOLOv8 variant  (default: yolov8s)")
    p.add_argument("--epochs",  type=int,   default=100)
    p.add_argument("--batch",   type=int,   default=16)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--device",  default="",
                   help="Training device: '' = auto, 'cpu', '0' (GPU 0), '0,1' (multi-GPU)")
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name",    default="f1_detector")
    p.add_argument("--resume",  action="store_true",
                   help="Resume interrupted training from last checkpoint")
    p.add_argument("--smoke",   action="store_true",
                   help="Quick smoke-test: 5 epochs, no augmentation")

    # Evaluation
    p.add_argument("--weights", default="models/car_detector.pt",
                   help="Weights path for --eval-only")
    p.add_argument("--split",   default="test",
                   choices=["train","val","test"],
                   help="Dataset split to evaluate on  (default: test)")
    p.add_argument("--eval-conf", type=float, default=0.25,
                   help="Confidence threshold for evaluation  (default: 0.25)")
    p.add_argument("--eval-iou",  type=float, default=0.45,
                   help="IoU threshold for evaluation  (default: 0.45)")
    p.add_argument("--no-report", action="store_true",
                   help="Skip saving the text report to file")

    # Keypoint CNN
    p.add_argument("--keypoint-annotations", default="dataset/keypoints/",
                   help="Directory of keypoint annotation JSONs")
    p.add_argument("--keypoint-circuit",     default="silverstone",
                   help="Circuit name for keypoint CNN  (default: silverstone)")
    p.add_argument("--keypoint-epochs",      type=int, default=50)
    p.add_argument("--keypoint-save",        default="models/keypoint_cnn.pth")

    return p.parse_args()


def main():
    args = parse_args()

    if args.verify:
        ok = verify_dataset(args.data)
        sys.exit(0 if ok else 1)

    elif args.eval_only:
        evaluate(args.weights, args.data, args.imgsz, args.split,
                 save_report=not args.no_report,
                 conf=args.eval_conf, iou=args.eval_iou)

    elif args.train_keypoint:
        train_keypoint(
            annotations_dir = args.keypoint_annotations,
            circuit         = args.keypoint_circuit,
            epochs          = args.keypoint_epochs,
            save_path       = args.keypoint_save,
        )

    else:
        # verify -> train -> evaluate
        print("Running dataset verification before training …")
        ok = verify_dataset(args.data)
        if not ok and not args.smoke:
            print("[ERROR] Dataset has issues. Fix them or re-run with --smoke to ignore.")
            sys.exit(1)

        train(args)

        # Auto-evaluate on test split after training
        best = Path("models/car_detector.pt")
        if best.exists():
            print("\nAuto-evaluating on test split …")
            evaluate(str(best), args.data, args.imgsz, split="test",
                     save_report=True,
                     conf=args.eval_conf, iou=args.eval_iou)


if __name__ == "__main__":
    main()
