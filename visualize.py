"""
F1 Dataset — Class-Filtered Bounding Box Visualizer
=====================================================
Search your dataset by class name or ID and display all matching images
with their bounding boxes drawn. Results are shown in an OpenCV window
and optionally saved as a contact-sheet grid image.

Usage:
    # Show all crash images
    python visualize_class.py --class crash

    # Show all Ferrari images
    python visualize_class.py --class Ferrari

    # Show multiple classes at once
    python visualize_class.py --class crash penalty_car

    # Show class 12 (crash) by ID
    python visualize_class.py --class-id 12

    # Save results as a grid image instead of opening a window
    python visualize_class.py --class crash --save

    # Search val split, limit to 20 images
    python visualize_class.py --class pitstop --split val --max 20

    # List all available classes in the dataset
    python visualize_class.py --list-classes

Requirements:
    pip install opencv-python numpy
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  CLASS SCHEMA  — must match your data.yaml
# ══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES: dict[int, str] = {
    0:  "car",
    1:  "RedBull",
    2:  "Mercedes",
    3:  "Ferrari",
    4:  "McLaren",
    5:  "Alpine",
    6:  "AstonMartin",
    7:  "Williams",
    8:  "Haas",
    9:  "KickSauber",
    10: "RacingBulls",
    11: "track_surface",
    12: "crash",
    13: "penalty_car",
    14: "pitstop",
    15: "race_start",
    16: "marshal",
    17: "yellow_flag",
    18: "safety_car",
}

# Visually distinct BGR colours per class
CLASS_COLORS: dict[int, tuple] = {
    0:  (200, 200, 200),  # car          — white/grey
    1:  (255,  30,  30),  # RedBull      — blue
    2:  (180, 210, 180),  # Mercedes     — silver-green
    3:  (0,    0,  220),  # Ferrari      — red
    4:  (0,  140, 255),   # McLaren      — orange
    5:  (220,  80, 160),  # Alpine       — pink-blue
    6:  (0,  160,  80),   # AstonMartin  — racing green
    7:  (200, 170, 255),  # Williams     — light blue
    8:  (30,   30,  30),  # Haas         — dark
    9:  (50,  200,  50),  # KickSauber   — green
    10: (100, 100, 220),  # RacingBulls  — dark blue
    11: (50,  200, 100),  # track_surface— teal
    12: (0,    0,  255),  # crash        — bright red
    13: (0,   50,  255),  # penalty_car  — red-orange
    14: (255, 200,   0),  # pitstop      — yellow
    15: (0,  255,  255),  # race_start   — cyan
    16: (255, 128,   0),  # marshal      — orange
    17: (0,  220,  255),  # yellow_flag  — yellow
    18: (255, 255, 255),  # safety_car   — white
}


def get_color(cls_id: int) -> tuple:
    return CLASS_COLORS.get(cls_id, (0, 255, 0))


def get_name(cls_id: int) -> str:
    return CLASS_NAMES.get(cls_id, f"class_{cls_id}")


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA.YAML  (overrides CLASS_NAMES if present)
# ══════════════════════════════════════════════════════════════════════════════

def load_yaml_classes(dataset_dir: Path) -> None:
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
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  RESOLVE CLASS NAMES → IDs
# ══════════════════════════════════════════════════════════════════════════════

def resolve_class_ids(names: list[str]) -> list[int]:
    """
    Convert a list of class name strings to integer IDs.
    Case-insensitive. Raises SystemExit if any name is not found.
    """
    name_to_id = {v.lower(): k for k, v in CLASS_NAMES.items()}
    ids = []
    for n in names:
        n_lower = n.lower()
        if n_lower in name_to_id:
            ids.append(name_to_id[n_lower])
        else:
            print(f"\n  [ERROR] Class '{n}' not found in schema.")
            print(f"  Available classes:")
            for cid, cname in sorted(CLASS_NAMES.items()):
                print(f"    {cid:>3}  {cname}")
            sys.exit(1)
    return ids


# ══════════════════════════════════════════════════════════════════════════════
#  SCAN DATASET — find images containing target classes
# ══════════════════════════════════════════════════════════════════════════════

def find_images_with_classes(
    dataset_dir: Path,
    target_ids:  set[int],
    split:       str = "train",
    max_images:  int | None = None,
) -> list[tuple[Path, Path, list[dict]]]:
    """
    Scan split labels for boxes belonging to target_ids.

    Returns list of (image_path, label_path, matching_boxes) where
    matching_boxes is a list of {"cls", "cx", "cy", "w", "h"} dicts
    for boxes that belong to a target class (all other boxes also loaded
    for context but flagged differently).
    """
    # Locate images dir
    candidates = [
        dataset_dir / split / "images",
        dataset_dir / "images",
        dataset_dir / split,
    ]
    img_dir = next((c for c in candidates if c.exists()), None)
    if img_dir is None:
        print(f"  [ERROR] No images directory found for split '{split}' in {dataset_dir}")
        sys.exit(1)

    lbl_dir_candidates = [
        img_dir.parent / "labels",
        img_dir.parent.parent / split / "labels",
    ]
    lbl_dir = next((d for d in lbl_dir_candidates if d.exists()), None)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    results  = []

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in img_exts:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt") if lbl_dir else None
        if lbl_path is None or not lbl_path.exists():
            continue

        all_boxes     = []
        matching_boxes = []

        for line in lbl_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                box = {"cls": cls, "cx": cx, "cy": cy, "w": w, "h": h,
                       "match": cls in target_ids}
                all_boxes.append(box)
                if cls in target_ids:
                    matching_boxes.append(box)
            except ValueError:
                continue

        if matching_boxes:
            results.append((img_path, lbl_path, all_boxes))
            if max_images and len(results) >= max_images:
                break

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  DRAW BOXES ON IMAGE
# ══════════════════════════════════════════════════════════════════════════════

def draw_boxes(
    image:       np.ndarray,
    boxes:       list[dict],
    target_ids:  set[int],
    show_all:    bool = True,
    fill_alpha:  float = 0.22,
) -> np.ndarray:
    """
    Draw bounding boxes on image.
    Target class boxes: solid colour + filled + label.
    Other class boxes: thin grey outline only (for spatial context).
    """
    out  = image.copy()
    H, W = out.shape[:2]

    # Draw non-target boxes first (background context, thin grey)
    if show_all:
        for box in boxes:
            if box["cls"] in target_ids:
                continue
            x1 = int((box["cx"] - box["w"] / 2) * W)
            y1 = int((box["cy"] - box["h"] / 2) * H)
            x2 = int((box["cx"] + box["w"] / 2) * W)
            y2 = int((box["cy"] + box["h"] / 2) * H)
            cv2.rectangle(out, (x1, y1), (x2, y2), (90, 90, 90), 1)

    # Draw target class boxes on top (highlighted)
    for box in boxes:
        if box["cls"] not in target_ids:
            continue

        color = get_color(box["cls"])
        name  = get_name(box["cls"])

        x1 = max(0, int((box["cx"] - box["w"] / 2) * W))
        y1 = max(0, int((box["cy"] - box["h"] / 2) * H))
        x2 = min(W - 1, int((box["cx"] + box["w"] / 2) * W))
        y2 = min(H - 1, int((box["cy"] + box["h"] / 2) * H))

        # Semi-transparent fill
        if x2 > x1 and y2 > y1:
            overlay = out.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, fill_alpha, out, 1 - fill_alpha, 0, out)

        # Solid border (thick)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label pill
        label      = f"{box['cls']}: {name}"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, min(0.7, W / 800))
        thickness  = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        pad = 3
        ly1 = y1 - th - pad * 2 - baseline
        if ly1 < 0:
            ly1 = y1 + 2
        ly2 = ly1 + th + pad * 2 + baseline
        cv2.rectangle(out, (x1, ly1), (x1 + tw + pad * 2, ly2), color, -1)
        cv2.putText(out, label, (x1 + pad, ly2 - baseline - pad),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out


def add_info_bar(
    image:      np.ndarray,
    img_path:   Path,
    boxes:      list[dict],
    target_ids: set[int],
    idx:        int,
    total:      int,
) -> np.ndarray:
    """Add a dark info bar at the bottom of the image."""
    H, W   = image.shape[:2]
    bar_h  = 40
    out    = np.zeros((H + bar_h, W, 3), dtype=np.uint8)
    out[:H] = image
    out[H:] = (28, 28, 28)

    n_match = sum(1 for b in boxes if b["cls"] in target_ids)
    names   = ", ".join(sorted({get_name(b["cls"]) for b in boxes if b["cls"] in target_ids}))
    line1   = f"  [{idx+1}/{total}]  {img_path.name}"
    line2   = f"  {n_match} matching box{'es' if n_match!=1 else ''}  ({names})   "
    line2  += "SPACE/→=next  ←=prev  S=save  A=toggle all boxes  Q=quit"

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, line1, (4, H + 14), font, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(out, line2, (4, H + 30), font, 0.36, (140, 140, 140), 1, cv2.LINE_AA)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE CONTACT SHEET GRID
# ══════════════════════════════════════════════════════════════════════════════

def save_grid(
    results:    list[tuple],
    target_ids: set[int],
    out_path:   Path,
    cols:       int = 4,
    cell:       int = 220,
) -> None:
    """Save all matching images as a contact-sheet grid with boxes drawn."""
    rows = math.ceil(len(results) / cols)
    grid = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    grid[:] = 20

    for i, (img_path, _, boxes) in enumerate(results):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = draw_boxes(img, boxes, target_ids, show_all=False)
        img = cv2.resize(img, (cell, cell))

        # Filename label at top
        cv2.rectangle(img, (0, 0), (cell, 16), (0, 0, 0), -1)
        cv2.putText(img, img_path.stem[:28], (3, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

        r, c = divmod(i, cols)
        grid[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = img

    # Header bar
    target_names = ", ".join(get_name(cid) for cid in sorted(target_ids))
    header = np.zeros((36, grid.shape[1], 3), dtype=np.uint8)
    header[:] = (40, 40, 40)
    cv2.putText(header, f"Class filter: {target_names}   |   {len(results)} images",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    final = np.vstack([header, grid])

    cv2.imwrite(str(out_path), final, [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(f"\n  Saved contact sheet → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE VIEWER
# ══════════════════════════════════════════════════════════════════════════════

def run_viewer(
    results:    list[tuple],
    target_ids: set[int],
    save_dir:   Path,
) -> None:
    if not results:
        print("  No images found for the requested class(es).")
        return

    WINDOW  = "F1 Class Viewer"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 600)

    idx      = 0
    show_all = True   # toggle other-class boxes on/off

    while True:
        img_path, _, boxes = results[idx]
        image    = cv2.imread(str(img_path))
        if image is None:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(image, "Cannot load image", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        annotated = draw_boxes(image, boxes, target_ids, show_all=show_all)
        frame     = add_info_bar(annotated, img_path, boxes, target_ids,
                                 idx, len(results))

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord(' '), 83, 3):      # SPACE / RIGHT
            idx = (idx + 1) % len(results)
        elif key in (81, 2):                 # LEFT
            idx = (idx - 1) % len(results)
        elif key in (ord('a'), ord('A')):    # toggle all boxes
            show_all = not show_all
        elif key in (ord('s'), ord('S')):    # save current frame
            save_dir.mkdir(parents=True, exist_ok=True)
            out = save_dir / f"annotated_{img_path.stem}.jpg"
            cv2.imwrite(str(out), frame)
            print(f"  Saved → {out}")

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
#  PRINT STATS
# ══════════════════════════════════════════════════════════════════════════════

def print_class_stats(dataset_dir: Path, split: str) -> None:
    """Print class distribution across the dataset."""
    candidates = [dataset_dir / split / "labels", dataset_dir / "labels"]
    lbl_dir = next((d for d in candidates if d.exists()), None)
    if not lbl_dir:
        print(f"  No labels found for split '{split}'")
        return

    counts: dict[int, int] = {}
    n_imgs  = 0
    for txt in sorted(lbl_dir.glob("*.txt")):
        lines = txt.read_text().strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                counts[cls] = counts.get(cls, 0) + 1
        if lines:
            n_imgs += 1

    total = sum(counts.values())
    print(f"\n  {'─'*52}")
    print(f"  Dataset: {dataset_dir}  |  Split: {split}")
    print(f"  Images with labels: {n_imgs}  |  Total boxes: {total}")
    print(f"\n  {'ID':<5} {'Class':<18} {'Boxes':>7} {'%':>7}  Bar")
    print(f"  {'─'*52}")
    for cls_id, count in sorted(counts.items()):
        name = get_name(cls_id)
        pct  = count / max(total, 1) * 100
        bar  = "█" * max(1, int(pct / 2))
        print(f"  {cls_id:<5} {name:<18} {count:>7} {pct:>6.1f}%  {bar}")
    print(f"  {'─'*52}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="F1 Dataset — Class-Filtered Bounding Box Visualizer",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python visualize_class.py --class crash
  python visualize_class.py --class Ferrari penalty_car
  python visualize_class.py --class-id 12
  python visualize_class.py --class pitstop --split val --max 30 --save
  python visualize_class.py --list-classes
        """
    )

    parser.add_argument("--dataset", "-d", default="roboflow_dataset",
                        help="Dataset root folder (default: roboflow_dataset/)")
    parser.add_argument("--split", "-s", default="train",
                        choices=["train", "val", "test"],
                        help="Which split to search (default: train)")
    parser.add_argument("--class", "-c", dest="class_names", nargs="+",
                        metavar="CLASS",
                        help="Class name(s) to filter (e.g. crash Ferrari pitstop)")
    parser.add_argument("--class-id", dest="class_ids", nargs="+", type=int,
                        metavar="ID",
                        help="Class ID(s) to filter (e.g. 12 13)")
    parser.add_argument("--max", "-m", type=int, default=None,
                        help="Max number of images to retrieve (default: all)")
    parser.add_argument("--save", action="store_true",
                        help="Save a contact-sheet grid image instead of opening a window")
    parser.add_argument("--save-dir", default="class_previews",
                        help="Folder for saved images (default: class_previews/)")
    parser.add_argument("--list-classes", action="store_true",
                        help="List all classes and their counts, then exit")
    parser.add_argument("--no-context", action="store_true",
                        help="Hide non-target class boxes (show only matching class)")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset folder not found: {dataset_dir}")
        sys.exit(1)

    load_yaml_classes(dataset_dir)

    # ── List classes mode ─────────────────────────────────────────────────────
    if args.list_classes:
        print_class_stats(dataset_dir, args.split)
        print("  All classes in schema:")
        for cid, name in sorted(CLASS_NAMES.items()):
            print(f"    {cid:>3}  {name}")
        return

    # ── Resolve target class IDs ──────────────────────────────────────────────
    target_ids: set[int] = set()
    if args.class_names:
        target_ids.update(resolve_class_ids(args.class_names))
    if args.class_ids:
        for cid in args.class_ids:
            if cid not in CLASS_NAMES:
                print(f"[ERROR] Class ID {cid} not in schema. Use --list-classes to see all.")
                sys.exit(1)
            target_ids.add(cid)

    if not target_ids:
        parser.print_help()
        print("\n  Please specify --class or --class-id\n")
        sys.exit(1)

    target_names = ", ".join(get_name(cid) for cid in sorted(target_ids))
    print(f"\n  Searching split='{args.split}' for class(es): {target_names}")

    # ── Scan dataset ──────────────────────────────────────────────────────────
    results = find_images_with_classes(
        dataset_dir = dataset_dir,
        target_ids  = target_ids,
        split       = args.split,
        max_images  = args.max,
    )

    if not results:
        print(f"\n  No images found containing: {target_names}")
        print(f"  Try --split val or --split test, or check --list-classes")
        sys.exit(0)

    print(f"  Found {len(results)} image(s) containing: {target_names}\n")

    save_dir = Path(args.save_dir)

    # ── Save grid or open interactive viewer ──────────────────────────────────
    if args.save:
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_name  = target_names.replace(", ", "_").replace(" ", "_")
        grid_path  = save_dir / f"grid_{safe_name}_{args.split}.jpg"
        save_grid(results, target_ids, grid_path)
        print(f"  Grid saved → {grid_path}")
    else:
        print("  Controls: SPACE/→=next  ←=prev  A=toggle context boxes  S=save frame  Q=quit\n")
        show_all = not args.no_context
        run_viewer(results, target_ids, save_dir)


if __name__ == "__main__":
    main()