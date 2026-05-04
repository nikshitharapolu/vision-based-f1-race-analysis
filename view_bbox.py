"""
F1 Dataset Bounding Box Viewer
================================
Visualise YOLO-format bounding boxes on your dataset images using OpenCV.

Controls:
  SPACE / RIGHT arrow  — next image
  LEFT arrow           — previous image
  1-9                  — filter by class ID (press same key again to clear)
  L                    — toggle labels on/off
  B                    — toggle filled boxes vs outline only
  S                    — save current annotated frame to output folder
  G                    — jump to a specific image number (type number + ENTER)
  Q / ESC              — quit

Requirements:
    pip install opencv-python numpy

Usage:
    python view_bbox.py                          # auto-detects roboflow_dataset/
    python view_bbox.py --dataset path/to/dataset
    python view_bbox.py --split val              # view val split instead of train
    python view_bbox.py --start 50               # start from image #50
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  CLASS SCHEMA — matches your unified data.yaml
# ══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = {
    0: "car",
    1: "penalty_car",
    2: "RedBull",
    3: "Alpine",
    4: "McLaren",
    5: "Mercedes",
    6: "AstonMartin",
    7: "Ferrari",
    8: "Williams",
}

# BGR colours per class (visually distinct)
CLASS_COLORS = {
    0: (200, 200, 200),   # car          — white/grey
    1: (0,   50,  255),   # penalty_car  — red
    2: (255, 50,    0),   # RedBull      — blue
    3: (255, 130,   0),   # Alpine       — pink/blue → orange
    4: (0,  140,  255),   # McLaren      — papaya orange
    5: (0,  210,  180),   # Mercedes     — silver-green
    6: (0,  150,   80),   # AstonMartin  — racing green
    7: (20,  20,  200),   # Ferrari      — red
    8: (200, 170, 255),   # Williams     — light blue/purple
}

def get_color(cls_id: int) -> tuple:
    return CLASS_COLORS.get(cls_id, (0, 255, 255))

def get_name(cls_id: int) -> str:
    return CLASS_NAMES.get(cls_id, f"class_{cls_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET LOADER
# ══════════════════════════════════════════════════════════════════════════════

def find_pairs(dataset_dir: Path, split: str) -> list[tuple[Path, Path | None]]:
    """
    Find all (image_path, label_path_or_None) pairs for a given split.
    Supports both:
      dataset/train/images/*.jpg  +  dataset/train/labels/*.txt
      dataset/images/*.jpg        +  dataset/labels/*.txt  (flat layout)
    """
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    # Try split subfolder first
    candidates = [
        dataset_dir / split / "images",
        dataset_dir / "images",
        dataset_dir / split,
        dataset_dir,
    ]
    img_dir = None
    for c in candidates:
        if c.exists() and any(f.suffix.lower() in img_exts for f in c.iterdir() if f.is_file()):
            img_dir = c
            break

    if img_dir is None:
        print(f"[ERROR] No images found in {dataset_dir} for split '{split}'")
        sys.exit(1)

    # Find matching labels dir
    lbl_dir_candidates = [
        img_dir.parent / "labels",
        img_dir.parent.parent / split / "labels",
        img_dir.parent / "labels",
    ]
    lbl_dir = next((d for d in lbl_dir_candidates if d.exists()), None)

    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in img_exts:
            continue
        lbl_path = None
        if lbl_dir:
            candidate_lbl = lbl_dir / (img_path.stem + ".txt")
            if candidate_lbl.exists():
                lbl_path = candidate_lbl
        pairs.append((img_path, lbl_path))

    return pairs


def load_label(lbl_path: Path) -> list[dict]:
    """Parse a YOLO .txt label file into a list of box dicts."""
    boxes = []
    if lbl_path is None or not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            boxes.append({
                "cls": int(parts[0]),
                "cx":  float(parts[1]),
                "cy":  float(parts[2]),
                "w":   float(parts[3]),
                "h":   float(parts[4]),
            })
        except ValueError:
            continue
    return boxes


def load_yaml_classes(dataset_dir: Path) -> None:
    """
    Optionally read class names from data.yaml and override CLASS_NAMES/CLASS_COLORS.
    Safe to skip if pyyaml not installed.
    """
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        return
    try:
        import yaml
        data  = yaml.safe_load(yaml_path.read_text())
        names = data.get("names", {})
        if isinstance(names, list):
            for i, n in enumerate(names):
                CLASS_NAMES[i] = n
        elif isinstance(names, dict):
            for k, v in names.items():
                CLASS_NAMES[int(k)] = v
        print(f"  Loaded {len(CLASS_NAMES)} class names from data.yaml")
    except Exception:
        pass   # pyyaml not installed or bad yaml — use defaults


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def draw_boxes(
    image:        np.ndarray,
    boxes:        list[dict],
    filter_cls:   int | None = None,
    show_labels:  bool       = True,
    filled:       bool       = True,
    fill_alpha:   float      = 0.25,
) -> np.ndarray:
    """Draw YOLO bounding boxes on a copy of image."""
    out  = image.copy()
    H, W = out.shape[:2]

    for box in boxes:
        cls = box["cls"]
        if filter_cls is not None and cls != filter_cls:
            continue

        color = get_color(cls)
        name  = get_name(cls)

        x1 = int((box["cx"] - box["w"] / 2) * W)
        y1 = int((box["cy"] - box["h"] / 2) * H)
        x2 = int((box["cx"] + box["w"] / 2) * W)
        y2 = int((box["cy"] + box["h"] / 2) * H)

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W-1, x2), min(H-1, y2)

        # Semi-transparent fill
        if filled and (x2 > x1) and (y2 > y1):
            overlay       = out.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, fill_alpha, out, 1 - fill_alpha, 0, out)

        # Solid border
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label pill
        if show_labels:
            label     = f"{cls}: {name}"
            font      = cv2.FONT_HERSHEY_SIMPLEX
            font_scale= 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            pad = 3
            # Place label above box, flip below if it would clip top
            lx1 = x1
            ly1 = y1 - th - pad * 2 - baseline
            if ly1 < 0:
                ly1 = y1 + 2
            lx2 = lx1 + tw + pad * 2
            ly2 = ly1 + th + pad * 2 + baseline

            cv2.rectangle(out, (lx1, ly1), (lx2, ly2), color, -1)
            # White text
            cv2.putText(out, label,
                        (lx1 + pad, ly2 - baseline - pad),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out


def build_hud(
    image:       np.ndarray,
    img_path:    Path,
    boxes:       list[dict],
    idx:         int,
    total:       int,
    filter_cls:  int | None,
    show_labels: bool,
    filled:      bool,
    split:       str,
) -> np.ndarray:
    """Overlay HUD info bar at the bottom of the image."""
    H, W = image.shape[:2]
    bar_h = 38
    hud   = np.zeros((H + bar_h, W, 3), dtype=np.uint8)
    hud[:H] = image

    # Bottom bar (dark)
    hud[H:] = (28, 28, 28)

    visible = [b for b in boxes if filter_cls is None or b["cls"] == filter_cls]

    # Count per class
    cls_counts: dict[int, int] = {}
    for b in visible:
        cls_counts[b["cls"]] = cls_counts.get(b["cls"], 0) + 1
    cls_str = "  ".join(f"{get_name(c)}×{n}" for c, n in sorted(cls_counts.items()))

    # Filter indicator
    flt_str = f"  [cls={filter_cls}: {get_name(filter_cls)}]" if filter_cls is not None else ""

    line1 = f"  [{idx+1}/{total}]  {img_path.name}{flt_str}"
    line2 = f"  Boxes: {len(visible)}   {cls_str}"

    font  = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(hud, line1, (4, H + 14), font, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(hud, line2, (4, H + 30), font, 0.38, (150, 150, 150), 1, cv2.LINE_AA)

    # Right side: mode indicators
    modes = []
    if show_labels: modes.append("Labels:ON")
    if filled:      modes.append("Fill:ON")
    mode_str = "  ".join(modes)
    cv2.putText(hud, mode_str, (W - 160, H + 14), font, 0.38, (100, 180, 100), 1, cv2.LINE_AA)

    # Shortcut reminder
    cv2.putText(hud, "SPC=next  L=labels  B=fill  S=save  G=goto  Q=quit",
                (W - 420, H + 30), font, 0.33, (90, 90, 90), 1, cv2.LINE_AA)

    return hud


# ══════════════════════════════════════════════════════════════════════════════
#  STATISTICS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_stats(pairs: list, split: str) -> None:
    total_imgs  = len(pairs)
    total_boxes = 0
    cls_counts: dict[int, int] = {}
    no_label    = 0

    for img_path, lbl_path in pairs:
        boxes = load_label(lbl_path)
        if not boxes:
            no_label += 1
        total_boxes += len(boxes)
        for b in boxes:
            cls_counts[b["cls"]] = cls_counts.get(b["cls"], 0) + 1

    print(f"\n{'─'*50}")
    print(f"  Dataset split : {split}")
    print(f"  Images        : {total_imgs}")
    print(f"  Total boxes   : {total_boxes}")
    print(f"  Avg per image : {total_boxes/max(total_imgs,1):.1f}")
    print(f"  No-label imgs : {no_label}")
    print(f"\n  Class breakdown:")
    for cls_id, count in sorted(cls_counts.items()):
        pct  = count / max(total_boxes, 1) * 100
        bar  = "█" * int(pct / 2)
        print(f"    {cls_id:>2}  {get_name(cls_id):<15}  {count:>5}  {pct:5.1f}%  {bar}")
    print(f"{'─'*50}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN VIEWER LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_viewer(
    dataset_dir: Path,
    split:       str  = "train",
    start_idx:   int  = 0,
    save_dir:    Path = Path("bbox_previews"),
) -> None:
    load_yaml_classes(dataset_dir)
    pairs = find_pairs(dataset_dir, split)

    if not pairs:
        print("[ERROR] No image/label pairs found.")
        sys.exit(1)

    print_stats(pairs, split)
    print(f"  Viewing {len(pairs)} images from '{split}' split")
    print(f"  Controls: SPACE/→=next  ←=prev  1-9=filter class  L=labels  B=fill  S=save  G=goto  Q=quit\n")

    save_dir.mkdir(parents=True, exist_ok=True)

    idx          = max(0, min(start_idx, len(pairs) - 1))
    filter_cls   = None
    show_labels  = True
    filled       = True
    goto_buffer  = ""   # accumulates digit keys for G command

    WINDOW = "F1 Dataset Viewer"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 600)

    while True:
        img_path, lbl_path = pairs[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [WARN] Could not read {img_path}")
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(image, "Image load error", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        boxes      = load_label(lbl_path)
        annotated  = draw_boxes(image, boxes, filter_cls, show_labels, filled)
        frame      = build_hud(annotated, img_path, boxes, idx, len(pairs),
                               filter_cls, show_labels, filled, split)

        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), ord('Q'), 27):   # Q or ESC
            break

        elif key in (ord(' '), 83, 3):        # SPACE or RIGHT arrow
            idx = (idx + 1) % len(pairs)

        elif key in (81, 2):                  # LEFT arrow
            idx = (idx - 1) % len(pairs)

        elif key in (ord('l'), ord('L')):
            show_labels = not show_labels

        elif key in (ord('b'), ord('B')):
            filled = not filled

        elif key in (ord('s'), ord('S')):     # Save current frame
            out_path = save_dir / f"annotated_{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), frame)
            print(f"  Saved → {out_path}")

        elif key in (ord('g'), ord('G')):     # Start goto mode
            goto_buffer = ""
            print("  Enter image number and press ENTER:")

        elif key == 13 and goto_buffer:       # ENTER — confirm goto
            try:
                target = int(goto_buffer) - 1   # 1-indexed input
                idx    = max(0, min(target, len(pairs) - 1))
                print(f"  Jumped to image {idx+1}")
            except ValueError:
                pass
            goto_buffer = ""

        elif 48 <= key <= 57 and goto_buffer != "":
            # Accumulate digits for goto
            goto_buffer += chr(key)
            print(f"  Goto: {goto_buffer}_", end="\r")

        elif ord('1') <= key <= ord('9') and goto_buffer == "":
            # Filter by class (press same key to clear)
            cls_id = key - ord('0') - 1      # '1' → 0, '2' → 1, etc.
            if filter_cls == cls_id:
                filter_cls = None
                print(f"  Filter cleared — showing all classes")
            else:
                filter_cls = cls_id
                print(f"  Filtering: class {cls_id} ({get_name(cls_id)})")

        elif key == ord('0') and goto_buffer == "":
            filter_cls = None
            print(f"  Filter cleared — showing all classes")

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    print("\n  Viewer closed.")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="F1 Dataset Bounding Box Viewer",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="roboflow_dataset",
        help="Path to dataset root folder (default: roboflow_dataset/)",
    )
    parser.add_argument(
        "--split", "-s",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split to view (default: train)",
    )
    parser.add_argument(
        "--start", "-n",
        type=int,
        default=0,
        help="Image index to start from, 0-based (default: 0)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="bbox_previews",
        help="Folder to save annotated frames (S key) (default: bbox_previews/)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print dataset statistics and exit without opening the viewer",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset folder not found: {dataset_dir}")
        print(f"  Run the viewer from the same directory as your dataset,")
        print(f"  or pass --dataset /full/path/to/your/dataset")
        sys.exit(1)

    load_yaml_classes(dataset_dir)

    if args.stats_only:
        pairs = find_pairs(dataset_dir, args.split)
        print_stats(pairs, args.split)
        return

    run_viewer(
        dataset_dir = dataset_dir,
        split       = args.split,
        start_idx   = args.start,
        save_dir    = Path(args.save_dir),
    )


if __name__ == "__main__":
    main()
