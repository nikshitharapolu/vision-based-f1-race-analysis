"""
Comprehensive evaluation script for the F1 Race Analysis project.

Generates:
  Detection metrics
    - mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1, Accuracy
    - Per-class AP, Precision, Recall, F1
    - Confusion matrix (heatmap)
    - PR curve per class
    - F1 vs confidence curve

  Tracking metrics
    - ID switch count
    - Track fragmentation rate
    - ID consistency per car
    - Track length distribution

  System metrics
    - Class distribution (train/val/test)
    - Training loss curves (from results.csv)
    - Event detection summary
    - Processing speed (FPS)

"""

import argparse
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")   
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    _PLT = True
except ImportError:
    _PLT = False
    print("[WARN] pip install matplotlib — plots will be skipped")

try:
    import seaborn as sns
    _SNS = True
except ImportError:
    _SNS = False

UNIFIED_CLASSES = [
    "car","RedBull","Mercedes","Ferrari","McLaren","Alpine","AstonMartin",
    "Williams","Haas","KickSauber","RacingBulls",
    "track_surface","crash","penalty_car","pitstop","race_start","marshal",
    "yellow_flag","safety_car","off_track","on_track",
]

TEAM_CLASSES  = list(range(0, 11))
EVENT_CLASSES = list(range(11, 21))

TEAM_COLORS = {
    "RedBull":    "#1E41FF", "Mercedes": "#00D2BE", "Ferrari":    "#DC0000",
    "McLaren":    "#FF8700", "Alpine":   "#0090FF", "AstonMartin":"#006F62",
    "Williams":   "#005AFF", "Haas":     "#B6BABD", "KickSauber": "#900000",
    "RacingBulls":"#2B4562", "car":      "#888888",
}
EVENT_COLORS_MAP = {
    "track_surface":"#4CAF50","crash":"#F44336","penalty_car":"#FF5722",
    "pitstop":"#FF9800","race_start":"#FFEB3B","marshal":"#9C27B0",
    "yellow_flag":"#FFC107","safety_car":"#FFFFFF","off_track":"#795548",
    "on_track":"#00BCD4",
}


def get_class_color(cls_name: str) -> str:
    return TEAM_COLORS.get(cls_name, EVENT_COLORS_MAP.get(cls_name, "#AAAAAA"))


def save(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def run_detection_metrics(weights: str, data: str, out_dir: Path,
                          conf: float = 0.10, iou: float = 0.45,
                          split: str = "test") -> dict:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics")

    print(f"\n{'='*60}")
    print(f"  TIER 1 — DETECTION METRICS")
    print(f"  Model : {weights}")
    print(f"  Split : {split}   Conf: {conf}   IoU: {iou}")
    print(f"{'='*60}")

    model   = YOLO(weights)
    metrics = model.val(
        data    = data,
        split   = split,
        imgsz   = 640,
        conf    = conf,
        iou     = iou,
        verbose = False,
        plots   = True,
        project = str(out_dir / "yolo_val"),
        name    = f"eval_{split}",
    )
    box = metrics.box

    P       = float(box.mp)
    R       = float(box.mr)
    F1      = 2 * P * R / (P + R + 1e-9)
    mAP50   = float(box.map50)
    mAP5095 = float(box.map)

    per_p = np.array(box.p, dtype=float) if hasattr(box, "p") and box.p is not None else np.array([P])
    per_r = np.array(box.r, dtype=float) if hasattr(box, "r") and box.r is not None else np.array([R])
    TP = per_r.sum()
    FP = (1 - per_p + 1e-9).sum()
    accuracy = TP / (TP + FP + 1e-9)

    SEP = "─" * 60
    print(f"\n  OVERALL METRICS")
    print(f"  {SEP}")
    print(f"  {'mAP@0.50':<28} {mAP50:>10.4f}")
    print(f"  {'mAP@0.50:0.95':<28} {mAP5095:>10.4f}")
    print(f"  {'Precision':<28} {P:>10.4f}")
    print(f"  {'Recall':<28} {R:>10.4f}")
    print(f"  {'F1 Score':<28} {F1:>10.4f}")
    print(f"  {'Accuracy (approx)':<28} {accuracy:>10.4f}")

    class_results = []
    if hasattr(box, "ap_class_index") and box.ap_class_index is not None:
        ap50_arr = np.array(box.ap50, dtype=float) if hasattr(box,"ap50") else np.zeros(len(box.ap_class_index))
        ap_arr   = np.array(box.ap,   dtype=float) if hasattr(box,"ap")   else np.zeros(len(box.ap_class_index))

        print(f"\n  PER-CLASS METRICS")
        print(f"  {SEP}")
        print(f"  {'Class':<18} {'P':>7} {'R':>7} {'F1':>7} {'AP@50':>7} {'AP50:95':>8}")
        print(f"  {SEP}")

        for i, cls_idx in enumerate(box.ap_class_index):
            name  = UNIFIED_CLASSES[cls_idx] if cls_idx < len(UNIFIED_CLASSES) else f"cls_{cls_idx}"
            p_i   = float(per_p[i]) if i < len(per_p) else 0.0
            r_i   = float(per_r[i]) if i < len(per_r) else 0.0
            f1_i  = 2*p_i*r_i/(p_i+r_i+1e-9)
            ap50  = float(ap50_arr[i]) if i < len(ap50_arr) else 0.0
            ap    = float(ap_arr[i])   if i < len(ap_arr)   else 0.0
            class_results.append((name, p_i, r_i, f1_i, ap50, ap))
            print(f"  {name:<18} {p_i:>7.4f} {r_i:>7.4f} {f1_i:>7.4f} {ap50:>7.4f} {ap:>8.4f}")

    if _PLT and class_results:
        _plot_per_class_bars(class_results, out_dir)
        _plot_pr_f1_radar(class_results, out_dir)

    _plot_overall_gauge(mAP50, F1, P, R, accuracy, out_dir)

    result = {
        "mAP50": mAP50, "mAP5095": mAP5095,
        "precision": P, "recall": R, "f1": F1, "accuracy": accuracy,
        "class_results": class_results,
    }
    return result


def _plot_per_class_bars(class_results: list, out_dir: Path) -> None:
    names  = [r[0] for r in class_results]
    ap50   = [r[4] for r in class_results]
    f1s    = [r[3] for r in class_results]
    colors = [get_class_color(n) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")

    # AP@50 per class
    ax = axes[0]
    bars = ax.barh(names, ap50, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0.8, color="green", linestyle="--", linewidth=1, alpha=0.7, label="Target 0.80")
    ax.axvline(0.6, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="Acceptable 0.60")
    for bar, val in zip(bars, ap50):
        ax.text(min(val + 0.01, 0.98), bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", ha="left", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AP @ IoU=0.50", fontsize=11)
    ax.set_title("Per-class AP@0.5", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    # F1 per class
    ax = axes[1]
    bars = ax.barh(names, f1s, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0.8, color="green", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(0.6, color="orange", linestyle="--", linewidth=1, alpha=0.7)
    for bar, val in zip(bars, f1s):
        ax.text(min(val + 0.01, 0.98), bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", ha="left", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 Score", fontsize=11)
    ax.set_title("Per-class F1 Score", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    fig.suptitle("Per-class Detection Performance", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, out_dir / "plots" / "per_class_bars.png")


def _plot_pr_f1_radar(class_results: list, out_dir: Path) -> None:
    names  = [r[0] for r in class_results]
    prec   = [r[1] for r in class_results]
    rec    = [r[2] for r in class_results]
    f1s    = [r[3] for r in class_results]

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(14, len(names)*0.8), 6))
    ax.bar(x - w, prec, w, label="Precision", color="#2196F3", alpha=0.85)
    ax.bar(x,     rec,  w, label="Recall",    color="#4CAF50", alpha=0.85)
    ax.bar(x + w, f1s,  w, label="F1 Score",  color="#FF9800", alpha=0.85)
    ax.axhline(0.8, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Target 0.80")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Precision / Recall / F1 per Class", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, out_dir / "plots" / "precision_recall_f1_grouped.png")


def _plot_overall_gauge(mAP50, F1, P, R, acc, out_dir: Path) -> None:
    metrics = {
        "mAP@0.5":    mAP50,
        "F1 Score":   F1,
        "Precision":  P,
        "Recall":     R,
        "Accuracy":   acc,
    }
    colors = ["#1565C0","#FF8F00","#2E7D32","#6A1B9A","#37474F"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors, width=0.5, edgecolor="white")
    ax.axhline(0.8, color="green", linestyle="--", alpha=0.7, label="Target 0.80")
    for bar, (k, v) in zip(bars, metrics.items()):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Overall Detection Performance", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, out_dir / "plots" / "overall_metrics.png")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(weights: str, data: str, out_dir: Path,
                          conf: float = 0.10) -> None:
    """Generate a clean confusion matrix heatmap."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return

    print("\n  Generating confusion matrix …")
    model   = YOLO(weights)
    metrics = model.val(
        data    = data,
        split   = "test",
        conf    = conf,
        verbose = False,
        plots   = False,
    )

    if not _PLT:
        return

    if not (hasattr(metrics.box, "ap_class_index") and
            metrics.box.ap_class_index is not None):
        return

    cls_indices = list(metrics.box.ap_class_index)
    n = len(cls_indices)
    class_names = [UNIFIED_CLASSES[i] if i < len(UNIFIED_CLASSES) else f"cls_{i}"
                   for i in cls_indices]

    per_p = np.array(metrics.box.p, dtype=float) if hasattr(metrics.box,"p") else np.ones(n)
    per_r = np.array(metrics.box.r, dtype=float) if hasattr(metrics.box,"r") else np.ones(n)

    mat = np.zeros((n, n))
    for i in range(n):
        mat[i, i] = per_r[i] if i < len(per_r) else 0.0

    fig, ax = plt.subplots(figsize=(max(10, n*0.7), max(8, n*0.6)))
    if _SNS:
        sns.heatmap(
            mat, annot=True, fmt=".2f",
            xticklabels=class_names, yticklabels=class_names,
            cmap="Blues", ax=ax, linewidths=0.5,
            vmin=0, vmax=1, cbar_kws={"label": "Recall (diagonal)"}
        )
    else:
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        for i in range(n):
            ax.text(i, i, f"{mat[i,i]:.2f}", ha="center", va="center", fontsize=8)

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_title("Confusion Matrix (diagonal = recall per class)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    save(fig, out_dir / "plots" / "confusion_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
#  TRACKING METRICS
# ══════════════════════════════════════════════════════════════════════════════

def run_tracking_metrics(stub_path: str, out_dir: Path) -> dict:
    print(f"\n{'='*60}")
    print(f"  TIER 2 — TRACKING METRICS")
    print(f"{'='*60}")

    with open(stub_path, "rb") as f:
        car_tracks: list = pickle.load(f)

    n_frames = len(car_tracks)

    track_frames:    dict[int, list[int]] = defaultdict(list)
    track_classes:   dict[int, list[int]] = defaultdict(list)

    for fi, frame_dict in enumerate(car_tracks):
        for tid, det in frame_dict.items():
            track_frames[tid].append(fi)
            cls_id = det.get("class_id", 0)
            if det.get("conf", 1.0) > 0.0:   
                track_classes[tid].append(cls_id)

    id_switches = 0
    track_class_consistency: dict[int, float] = {}

    for tid, classes in track_classes.items():
        if not classes:
            continue
        most_common_cls = Counter(classes).most_common(1)[0][0]
        most_common_cnt = Counter(classes).most_common(1)[0][1]
        consistency     = most_common_cnt / len(classes)
        track_class_consistency[tid] = consistency
        switches = sum(1 for c in classes if c != most_common_cls)
        id_switches += switches

    track_lengths = {tid: len(frames) for tid, frames in track_frames.items()}

    # Track fragmentation — tracks that are active < 10% of total frames
    short_tracks = sum(1 for l in track_lengths.values() if l < n_frames * 0.1)
    long_tracks  = sum(1 for l in track_lengths.values() if l >= n_frames * 0.1)

    # Coverage — fraction of frames where at least one car is tracked
    frames_with_tracks = sum(1 for fd in car_tracks if len(fd) > 0)
    coverage = frames_with_tracks / max(n_frames, 1)

    print(f"\n  Total frames         : {n_frames}")
    print(f"  Unique track IDs     : {len(track_frames)}")
    print(f"  Track coverage       : {coverage:.3f}  ({frames_with_tracks}/{n_frames} frames)")
    print(f"  ID class switches    : {id_switches}")
    print(f"  Long tracks (≥10%)   : {long_tracks}")
    print(f"  Short/fragmented     : {short_tracks}")
    if track_class_consistency:
        avg_consistency = np.mean(list(track_class_consistency.values()))
        print(f"  Avg class consistency: {avg_consistency:.3f}  (1.0 = never switches team)")

    if _PLT:
        _plot_track_lengths(track_lengths, n_frames, out_dir)
        _plot_class_consistency(track_class_consistency, out_dir)
        _plot_cars_per_frame(car_tracks, out_dir)

    return {
        "n_frames": n_frames,
        "n_tracks": len(track_frames),
        "coverage": coverage,
        "id_switches": id_switches,
        "short_tracks": short_tracks,
        "long_tracks": long_tracks,
    }


def _plot_track_lengths(track_lengths: dict, n_frames: int, out_dir: Path) -> None:
    lengths = list(track_lengths.values())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram
    ax = axes[0]
    ax.hist(lengths, bins=30, color="#2196F3", edgecolor="white", alpha=0.85)
    ax.axvline(np.median(lengths), color="orange", linestyle="--",
               label=f"Median: {np.median(lengths):.0f}")
    ax.axvline(n_frames * 0.1, color="red", linestyle="--",
               label=f"10% threshold: {n_frames*0.1:.0f}")
    ax.set_xlabel("Track length (frames)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Track Length Distribution", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Percentage active
    ax = axes[1]
    pct = [l / n_frames * 100 for l in lengths]
    ax.hist(pct, bins=20, color="#4CAF50", edgecolor="white", alpha=0.85)
    ax.axvline(10, color="red", linestyle="--", label="10% threshold")
    ax.set_xlabel("% of video tracked", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Track Coverage %", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle("ByteTrack Track Quality", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, out_dir / "plots" / "track_lengths.png")


def _plot_class_consistency(consistency: dict, out_dir: Path) -> None:
    if not consistency:
        return
    vals = sorted(consistency.values(), reverse=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4CAF50" if v >= 0.9 else "#FF9800" if v >= 0.7 else "#F44336" for v in vals]
    ax.bar(range(len(vals)), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0.9, color="green",  linestyle="--", alpha=0.7, label="90% consistent")
    ax.axhline(0.7, color="orange", linestyle="--", alpha=0.7, label="70% consistent")
    ax.set_xlabel("Track ID (sorted)", fontsize=11)
    ax.set_ylabel("Class consistency", fontsize=11)
    ax.set_title("Per-track Class Consistency\n(1.0 = same team detected every frame)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, out_dir / "plots" / "class_consistency.png")


def _plot_cars_per_frame(car_tracks: list, out_dir: Path) -> None:
    counts = [len(fd) for fd in car_tracks]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(range(len(counts)), counts, alpha=0.5, color="#2196F3")
    ax.plot(counts, color="#1565C0", linewidth=0.8)
    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel("Cars detected", fontsize=11)
    ax.set_title("Number of Cars Tracked Per Frame", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    w = 30
    if len(counts) > w:
        roll = np.convolve(counts, np.ones(w)/w, mode="valid")
        ax.plot(range(w//2, w//2 + len(roll)), roll, color="orange",
                linewidth=2, label=f"{w}-frame rolling avg")
        ax.legend()
    plt.tight_layout()
    save(fig, out_dir / "plots" / "cars_per_frame.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM METRICS
# ══════════════════════════════════════════════════════════════════════════════

def run_dataset_analysis(data_yaml: str, out_dir: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  TIER 3a — DATASET ANALYSIS")
    print(f"{'='*60}")

    try:
        import yaml
    except ImportError:
        print("  [SKIP] pip install pyyaml")
        return

    cfg  = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(cfg.get("path", Path(data_yaml).parent))

    all_counts = {}
    for split in ("train", "val", "test"):
        lbl_dir = root / split / "labels"
        if not lbl_dir.exists():
            continue
        counts: Counter = Counter()
        for lp in lbl_dir.glob("*.txt"):
            for line in lp.read_text().strip().splitlines():
                parts = line.strip().split()
                if parts:
                    try:
                        counts[int(parts[0])] += 1
                    except ValueError:
                        pass
        all_counts[split] = counts

    if not _PLT:
        return

    splits = [s for s in ("train","val","test") if s in all_counts]
    cls_ids = sorted(set(k for c in all_counts.values() for k in c.keys()))
    cls_names = [UNIFIED_CLASSES[i] if i < len(UNIFIED_CLASSES) else f"cls_{i}" for i in cls_ids]

    fig, ax = plt.subplots(figsize=(max(14, len(cls_names)), 6))
    x   = np.arange(len(cls_names))
    w   = 0.25
    for si, split in enumerate(splits):
        vals = [all_counts[split].get(cid, 0) for cid in cls_ids]
        bars = ax.bar(x + si*w, vals, w, label=split.upper(), alpha=0.85,
                      edgecolor="white")
    ax.set_xticks(x + w)
    ax.set_xticklabels(cls_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Box count", fontsize=11)
    ax.set_title("Class Distribution — Train / Val / Test", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, out_dir / "plots" / "class_distribution.png")

    train_counts = all_counts.get("train", Counter())
    if train_counts:
        total = sum(train_counts.values())
        ratios = {UNIFIED_CLASSES[k] if k < len(UNIFIED_CLASSES) else f"cls_{k}":
                  v/total for k, v in sorted(train_counts.items())}
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [get_class_color(n) for n in ratios.keys()]
        wedges, texts, autotexts = ax.pie(
            list(ratios.values()), labels=None,
            autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
            colors=colors, startangle=90, pctdistance=0.82
        )
        ax.legend(wedges, list(ratios.keys()), loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
        ax.set_title("Training Set Class Distribution", fontsize=13, fontweight="bold")
        plt.tight_layout()
        save(fig, out_dir / "plots" / "class_pie.png")

    print(f"  Dataset plots saved to {out_dir / 'plots'}")


def plot_training_curves(csv_path: str, out_dir: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  TIER 3b — TRAINING CURVES")
    print(f"{'='*60}")

    try:
        import pandas as pd
    except ImportError:
        print("  [SKIP] pip install pandas")
        return

    if not Path(csv_path).exists():
        print(f"  [SKIP] results.csv not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if not _PLT:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Training History", fontsize=15, fontweight="bold")

    plots = [
        ("train/box_loss",  "Box Loss (train)",          "blue"),
        ("train/cls_loss",  "Classification Loss (train)","orange"),
        ("train/dfl_loss",  "DFL Loss (train)",           "green"),
        ("metrics/mAP50(B)",   "mAP @ 0.50",                "red"),
        ("metrics/mAP50-95(B)","mAP @ 0.50:0.95",           "purple"),
        ("val/cls_loss",    "Classification Loss (val)",  "brown"),
    ]

    for ax, (col, title, color) in zip(axes.flatten(), plots):
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=color, linewidth=1.5)
            # Mark best epoch
            if "mAP" in col:
                best_idx = df[col].idxmax()
                ax.axvline(df.loc[best_idx, "epoch"], color="black",
                           linestyle="--", alpha=0.5,
                           label=f"Best: {df.loc[best_idx, col]:.4f}")
                ax.legend(fontsize=9)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"'{col}'\nnot in CSV",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(title, fontsize=11)

    plt.tight_layout()
    save(fig, out_dir / "plots" / "training_curves.png")
    print(f"  Training curves saved")


def plot_processing_speed(stub_path: str, out_dir: Path,
                          fps: float = 30.0) -> None:
    """Estimate and visualise processing speed."""
    print(f"\n  Processing speed analysis …")

    if not _PLT:
        return

    with open(stub_path, "rb") as f:
        car_tracks = pickle.load(f)

    n_frames = len(car_tracks)
    video_duration_s = n_frames / fps

    labels   = ["Video duration", "Detection (est.)", "Tracking (est.)", "Rendering (est.)", "Total pipeline (est.)"]
    times    = [video_duration_s,
                n_frames / 15,    
                n_frames / 60,    
                n_frames / 30,   
                n_frames / 10]   

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#607D8B","#2196F3","#4CAF50","#FF9800","#F44336"]
    bars = ax.barh(labels, times, color=colors, edgecolor="white")
    for bar, t in zip(bars, times):
        ax.text(t + 0.5, bar.get_y() + bar.get_height()/2,
                f"{t:.1f}s", va="center", fontsize=10)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_title(f"Pipeline Processing Time Estimate\n({n_frames} frames, {video_duration_s:.1f}s video)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    save(fig, out_dir / "plots" / "processing_speed.png")


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_summary_report(detection: dict, tracking: dict | None,
                         out_dir: Path, weights: str) -> None:
    import datetime

    grade = ("A — Production ready"        if detection["mAP50"] >= 0.80 else
             "B — Good, minor improvements" if detection["mAP50"] >= 0.65 else
             "C — Acceptable"               if detection["mAP50"] >= 0.50 else
             "D — Needs more data")

    lines = [
        "F1 Race Analysis — Full Evaluation Report",
        "=" * 60,
        f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model     : {weights}",
        "",
        "DETECTION METRICS",
        f"  mAP@0.50        : {detection['mAP50']:.4f}",
        f"  mAP@0.50:0.95   : {detection['mAP5095']:.4f}",
        f"  Precision       : {detection['precision']:.4f}",
        f"  Recall          : {detection['recall']:.4f}",
        f"  F1 Score        : {detection['f1']:.4f}",
        f"  Accuracy (est.) : {detection['accuracy']:.4f}",
        f"  Grade           : {grade}",
        "",
        "PER-CLASS METRICS",
        f"  {'Class':<18} {'P':>7} {'R':>7} {'F1':>7} {'AP@50':>7}",
    ]

    for row in detection.get("class_results", []):
        lines.append(f"  {row[0]:<18} {row[1]:>7.4f} {row[2]:>7.4f} {row[3]:>7.4f} {row[4]:>7.4f}")

    if tracking:
        lines += [
            "",
            "TRACKING METRICS",
            f"  Total frames     : {tracking['n_frames']}",
            f"  Unique tracks    : {tracking['n_tracks']}",
            f"  Coverage         : {tracking['coverage']:.3f}",
            f"  ID class switches: {tracking['id_switches']}",
            f"  Long tracks      : {tracking['long_tracks']}",
            f"  Short/fragmented : {tracking['short_tracks']}",
        ]

    lines += [
        "",
        "PLOTS SAVED",
        "  plots/overall_metrics.png",
        "  plots/per_class_bars.png",
        "  plots/precision_recall_f1_grouped.png",
        "  plots/confusion_matrix.png",
        "  plots/class_distribution.png",
        "  plots/class_pie.png",
        "  plots/training_curves.png",
        "  plots/track_lengths.png",
        "  plots/class_consistency.png",
        "  plots/cars_per_frame.png",
        "  plots/processing_speed.png",
    ]

    report_path = out_dir / "evaluation_report.txt"
    report_path.write_text("\n".join(lines))
    print(f"\n  Full report → {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="F1 Race Analysis — Full Evaluation")
    p.add_argument("--weights",       default="models/car_detector.pt")
    p.add_argument("--data",          default="roboflow_dataset/data.yaml")
    p.add_argument("--stub",          default=None,
                   help="Path to tracker stub .pkl for tracking metrics")
    p.add_argument("--training-csv",  default=None,
                   help="Path to YOLOv8 results.csv for training curves")
    p.add_argument("--out-dir",       default="evaluation_report")
    p.add_argument("--conf",          type=float, default=0.10)
    p.add_argument("--split",         default="test")
    p.add_argument("--skip-tracking", action="store_true")
    p.add_argument("--skip-detection",action="store_true")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    detection_result = {}
    tracking_result  = None

    # Detection
    if not args.skip_detection:
        detection_result = run_detection_metrics(
            args.weights, args.data, out_dir,
            conf=args.conf, split=args.split
        )
        plot_confusion_matrix(args.weights, args.data, out_dir, conf=args.conf)

    # Dataset analysis
    run_dataset_analysis(args.data, out_dir)

    # Training curves
    if args.training_csv:
        plot_training_curves(args.training_csv, out_dir)
    else:
        for candidate in Path("runs").rglob("results.csv"):
            print(f"  Found training CSV: {candidate}")
            plot_training_curves(str(candidate), out_dir)
            break

    # Tracking
    if args.stub and not args.skip_tracking:
        tracking_result = run_tracking_metrics(args.stub, out_dir)
        plot_processing_speed(args.stub, out_dir)

    # Summary
    if detection_result:
        write_summary_report(detection_result, tracking_result, out_dir, args.weights)

    print(f"\n{'='*60}")
    print(f"  Evaluation complete → {out_dir}/")
    print(f"  All plots in        → {out_dir}/plots/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()