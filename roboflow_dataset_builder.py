"""
Roboflow Dataset Builder — Full Project Download
==================================================
Paste a Roboflow Universe PROJECT URL (not an image URL):
    https://universe.roboflow.com/aio-project/formula-1-dkxin
    https://universe.roboflow.com/project-ops31/formula-1-sexm0
    https://universe.roboflow.com/my-workspace-l15km/f1-car-bdlpu

The script downloads the ENTIRE dataset from each project URL automatically:
  1. Hits the Roboflow API to find the project + latest version number
  2. Requests a YOLOv8 export ZIP for that version (images + labels bundled)
  3. Downloads and extracts the ZIP — already has bounding-box .txt files
  4. If no export ZIP is available → falls back to search_all() + source CDN
     download + YOLOv8 auto-label
  5. Merges multiple projects into one unified dataset
  6. Writes a clean data.yaml + preview_grid.jpg for QA

Requirements:
    pip install requests opencv-python numpy Pillow ultralytics

Usage:
    1. Set ROBOFLOW_API_KEY  (free at app.roboflow.com → Settings → Roboflow API)
    2. Add your project URLs to DATASET_URLS
    3. python roboflow_dataset_builder.py
"""

import io
import math
import os
import random
import re
import shutil
import time
import zipfile
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURE THESE TWO THINGS
# ══════════════════════════════════════════════════════════════════════════════

ROBOFLOW_API_KEY = "xr7N7tqQDUoVAND781A0"
# Free at: https://app.roboflow.com/settings/api

DATASET_URLS = [
    "https://universe.roboflow.com/aio-project/formula-1-dkxin",
    "https://universe.roboflow.com/project-ops31/formula-1-sexm0",
    # "https://universe.roboflow.com/my-workspace-l15km/f1-car-bdlpu",
    # Add as many project URLs as you like — they all get merged
]
# Global class mapping (label → unified ID)
CLASS_MAP = {}

def get_or_create_class_id(label: str) -> int:
    if label not in CLASS_MAP:
        CLASS_MAP[label] = len(CLASS_MAP)
    return CLASS_MAP[label]

# ══════════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR      = "roboflow_dataset"
EXPORT_FORMAT   = "yolov8"
IMG_SIZE        = 640
CONF_THRESHOLD  = 0.25
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15
SEED            = 42
REQUEST_TIMEOUT = 60
MAX_IMAGES      = None   # set e.g. 500 to cap per project; None = all

RF_API        = "https://api.roboflow.com"
RF_SOURCE_CDN = "https://source.roboflow.com"

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

try:
    import requests
except ImportError:
    raise ImportError("pip install requests")

try:
    import cv2
except ImportError:
    raise ImportError("pip install opencv-python")

try:
    import numpy as np
except ImportError:
    raise ImportError("pip install numpy")

try:
    from PIL import Image
except ImportError:
    raise ImportError("pip install Pillow")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_class_names_from_yaml(extract_dir: Path) -> dict:
    """
    Reads data.yaml inside Roboflow export and returns:
    {old_class_id: class_name}
    """
    yaml_files = list(extract_dir.rglob("data.yaml"))
    if not yaml_files:
        return {}

    try:
        import yaml
    except ImportError:
        print("  [WARN] pyyaml not installed, class names not loaded")
        return {}

    try:
        data = yaml.safe_load(yaml_files[0].read_text())
        names = data.get("names", {})
        return {int(k): v for k, v in names.items()}
    except Exception as e:
        print(f"  [WARN] Failed to read data.yaml: {e}")
        return {}

def parse_project_url(url: str):
    """
    Extract (workspace, project) from a Universe project URL.
    Works for both project URLs and image URLs (strips /images/... suffix).
    """
    m = re.search(r"universe\.roboflow\.com/([^/]+)/([^/?#]+)", url)
    if not m:
        print(f"  [WARN] Cannot parse URL: {url}")
        return None
    return m.group(1), m.group(2)


def api_get(path: str, params: dict = None) -> dict | None:
    p = dict(params or {})
    p["api_key"] = ROBOFLOW_API_KEY
    try:
        r = requests.get(f"{RF_API}{path}", params=p, timeout=REQUEST_TIMEOUT)
        if r.status_code == 401:
            print("  [ERROR] 401 Unauthorized — check ROBOFLOW_API_KEY")
            return None
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [WARN] API GET {path}: {e}")
        return None


def download_bytes(url: str, label: str = "") -> bytes | None:
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT,
                         headers={"User-Agent": "RoboflowDatasetBuilder/2.0"})
        r.raise_for_status()
        return r.content
    except Exception as e:
        print(f"  [WARN] Download failed {label}: {e}")
        return None


def bytes_to_bgr(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        try:
            pil = Image.open(io.BytesIO(data)).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return img


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD A — Export ZIP  (best: images + YOLO labels in one download)
# ══════════════════════════════════════════════════════════════════════════════

def get_latest_version(workspace: str, project: str) -> int | None:
    """Return the latest version number for a project, or None."""
    data = api_get(f"/{workspace}/{project}")
    print("data: ", data)
    if not data:
        return None

    # FIX: correct versions source (top-level, not project["versions"])
    versions = data.get("versions", [])

    if not isinstance(versions, list) or not versions:
        return None

    best = None
    for v in versions:
        vid = v.get("id", "")
        m   = re.search(r"/(\d+)$", str(vid))
        if m:
            n = int(m.group(1))
            if best is None or n > best:
                best = n

    return best


def get_export_zip_url(workspace: str, project: str, version: int) -> str | None:
    """Request a YOLOv8 export link. Returns URL string or None."""
    # FIX: direct download endpoint (more reliable)
    return f"https://api.roboflow.com/{workspace}/{project}/{version}/download?format={EXPORT_FORMAT}&api_key={ROBOFLOW_API_KEY}"


def collect_from_zip(extract_dir: Path) -> list[tuple[Path, Path | None]]:
    """
    Walk extracted Roboflow ZIP, return (image_path, label_path_or_None) pairs.
    Roboflow ZIP layout:
        train/images/*.jpg  train/labels/*.txt
        valid/images/*.jpg  valid/labels/*.txt
        test/images/*.jpg   test/labels/*.txt
    """
    pairs = []
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for img_path in sorted(extract_dir.rglob("*")):
        if img_path.suffix.lower() not in img_exts:
            continue
        if "preview" in img_path.name.lower():
            continue
        # Standard Roboflow layout: images/ and labels/ are siblings
        lbl_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
        if not lbl_path.exists():
            lbl_path = img_path.with_suffix(".txt")
        pairs.append((img_path, lbl_path if lbl_path.exists() else None))
    return pairs


def load_yolo_label(lbl_path: Path, class_id_to_name: dict) -> list:
    """Read YOLO labels and convert to unified class IDs."""
    boxes = []
    for line in lbl_path.read_text().strip().splitlines():
        pts = line.strip().split()
        if len(pts) >= 5:
            try:
                old_cls = int(pts[0])
                cx = float(pts[1])
                cy = float(pts[2])
                w  = float(pts[3])
                h  = float(pts[4])

                class_name = class_id_to_name.get(old_cls, "unknown")
                new_cls = get_or_create_class_id(class_name)

                boxes.append((new_cls, cx, cy, w, h))
            except:
                continue
    return boxes


def try_zip_export(workspace: str, project: str, tmp: Path) -> list | None:
    """
    Attempt Method A (ZIP export).
    Returns list of (bgr_img, yolo_boxes, tag) or None if unavailable.
    """
    version = get_latest_version(workspace, project)
    if version is None:
        print(f"    No versions found for {workspace}/{project}")
        return None

    print(f"    Requesting YOLOv8 export (version {version}) …")
    zip_url = get_export_zip_url(workspace, project, version)
    if not zip_url:
        print(f"    Export URL not available for version {version}")
        return None

    print(f"    Downloading dataset ZIP …")
    zip_data = download_bytes(zip_url, f"{workspace}/{project} v{version}")
    if not zip_data:
        return None

    zip_dir = tmp / f"{workspace}_{project}" / "zip"
    zip_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            zf.extractall(zip_dir)
    except zipfile.BadZipFile as e:
        print(f"    Bad ZIP file: {e}")
        return None

    pairs = collect_from_zip(zip_dir)
    print(f"    ZIP contains {len(pairs)} images")
    class_id_to_name = load_class_names_from_yaml(zip_dir)
    if not pairs:
        return None

    records = []
    for img_path, lbl_path in pairs:
        if MAX_IMAGES and len(records) >= MAX_IMAGES:
            break
        img = cv2.imread(str(img_path))
        if img is None:
            data = img_path.read_bytes()
            img  = bytes_to_bgr(data)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        if lbl_path and lbl_path.exists():
            boxes = load_yolo_label(lbl_path, class_id_to_name)
            tag   = "zip_yolo_labels"
        else:
            boxes = auto_label(img)
            tag   = "yolo_autolabel"

    return records if records else None


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD B — search_all + source CDN fallback
# ══════════════════════════════════════════════════════════════════════════════

def search_all_images(workspace: str, project: str) -> list[dict]:
    """Page through search API to get all image records."""
    records = []
    offset  = 0
    limit   = 200

    print(f"    Fetching image list via search API …")
    while True:
        # Try POST search (most reliable)
        try:
            r = requests.post(
                f"{RF_API}/{workspace}/{project}/search",
                params={"api_key": ROBOFLOW_API_KEY},
                json={
                    "offset":     offset,
                    "limit":      limit,
                    "in_dataset": True,
                    "fields":     ["id", "name", "owner", "annotations"],
                },
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
        except Exception:
            # Fallback to GET
            data = api_get(f"/{workspace}/{project}/search",
                           {"offset": offset, "limit": limit,
                            "in_dataset": "true",
                            "fields": "id,name,owner,annotations"})
            if not data:
                break

        results = data.get("results", [])
        if not results:
            break

        records.extend(results)
        total = data.get("total", 0)
        print(f"    … {len(records)}/{total} image records fetched")

        if MAX_IMAGES and len(records) >= MAX_IMAGES:
            records = records[:MAX_IMAGES]
            break
        if len(records) >= total or len(results) < limit:
            break
        offset += limit

    return records


def fetch_image_via_cdn(record: dict, workspace: str, project: str) -> tuple:
    """Download image + annotations for one record. Returns (img, boxes)."""
    owner    = record.get("owner", workspace)
    image_id = record["id"]

    # Download from official source CDN pattern
    img_url  = f"{RF_SOURCE_CDN}/{owner}/{image_id}/original.jpg"
    img_data = download_bytes(img_url, image_id[:16])
    if not img_data:
        return None, []

    img = bytes_to_bgr(img_data)
    if img is None:
        return None, []

    # Fetch annotations
    boxes = []
    det = api_get(f"/{workspace}/{project}/images/{image_id}")
    if det:
        img_info   = det.get("image", {})
        annotation = img_info.get("annotation") or {}
        raw_boxes  = annotation.get("boxes", [])
        # img_w      = annotation.get("width",  img.shape[1])
        # img_h      = annotation.get("height", img.shape[0])
        try:
            img_w = float(annotation.get("width",  img.shape[1]))
            img_h = float(annotation.get("height", img.shape[0]))
        except:
            img_h, img_w = img.shape[:2]
        for b in raw_boxes:
            # cx = b.get("x", 0) / img_w
            # cy = b.get("y", 0) / img_h
            # bw = b.get("width",  0) / img_w
            # bh = b.get("height", 0) / img_h
            try:
                x  = float(b.get("x", 0))
                y  = float(b.get("y", 0))
                w  = float(b.get("width", 0))
                h  = float(b.get("height", 0))

                cx = x / img_w
                cy = y / img_h
                bw = w / img_w
                bh = h / img_h

            except (ValueError, TypeError):
                continue
            if bw > 0.005 and bh > 0.005:
                label = b.get("label", "unknown")
                cls_id = get_or_create_class_id(label)
                boxes.append((cls_id, cx, cy, bw, bh))
                # boxes.append((max(0., min(1., cx)), max(0., min(1., cy)),
                #                max(0.001, min(1., bw)), max(0.001, min(1., bh))))
    return img, boxes


def try_cdn_fallback(workspace: str, project: str) -> list:
    """Method B: search_all + CDN download + annotation API."""
    image_records = search_all_images(workspace, project)
    if not image_records:
        return []

    records = []
    total   = len(image_records)
    for i, rec in enumerate(image_records):
        img, boxes = fetch_image_via_cdn(rec, workspace, project)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        if not boxes:
            boxes = auto_label(img)
            tag   = "yolo_autolabel"
        else:
            tag = "api_annotations"
        records.append((img, boxes, tag))
        if (i+1) % 25 == 0 or (i+1) == total:
            print(f"    … {i+1}/{total} images downloaded")

    return records


# ══════════════════════════════════════════════════════════════════════════════
#  YOLO AUTO-LABEL
# ══════════════════════════════════════════════════════════════════════════════

_yolo_model = None

def auto_label(image: np.ndarray) -> list:
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            print("\n  Loading YOLOv8n for auto-label fallback …")
            _yolo_model = YOLO("yolov8n.pt")
        except ImportError:
            return []
    results = _yolo_model(image, conf=CONF_THRESHOLD, classes=[2, 7], verbose=False)
    H, W    = image.shape[:2]
    boxes   = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append((
                max(0., min(1., ((x1+x2)/2)/W)),
                max(0., min(1., ((y1+y2)/2)/H)),
                max(0.001, min(1., (x2-x1)/W)),
                max(0.001, min(1., (y2-y1)/H)),
            ))
    return boxes


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET WRITER
# ══════════════════════════════════════════════════════════════════════════════

def setup_dirs(out: Path) -> dict:
    dirs = {}
    for s in ("train", "val", "test"):
        (out/s/"images").mkdir(parents=True, exist_ok=True)
        (out/s/"labels").mkdir(parents=True, exist_ok=True)
        dirs[s] = (out/s/"images", out/s/"labels")
    return dirs


def assign_split(idx: int, total: int) -> str:
    r = idx / total
    if r < TRAIN_RATIO:                    return "train"
    if r < TRAIN_RATIO + VAL_RATIO:        return "val"
    return "test"


def write_label(path: Path, boxes: list) -> None:
    path.write_text(
        "\n".join(
            f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            for cls, cx, cy, bw, bh in boxes
        )
    )


def write_yaml(out: Path, sources: list) -> None:
    src_lines = "\n".join(f"#   {s}" for s in sources)

    names_block = "\n".join(
        f"  {i}: {name}" for name, i in CLASS_MAP.items()
    )

    (out/"data.yaml").write_text(f"""\
# F1 Car Detection — Roboflow Universe Dataset
# Sources:
{src_lines}

path: {out.resolve()}
train: train/images
val:   val/images
test:  test/images

nc: {len(CLASS_MAP)}
names:
{names_block}
""")


def save_preview(out: Path, n: int = 16, cols: int = 4, cell: int = 160) -> None:
    idir   = out/"train"/"images"
    ldir   = out/"train"/"labels"
    images = sorted(idir.glob("*.jpg"))[:n]
    if not images:
        return
    rows = math.ceil(len(images)/cols)
    grid = np.zeros((rows*cell, cols*cell, 3), dtype=np.uint8)
    grid[:] = 25
    for i, ip in enumerate(images):
        img = cv2.imread(str(ip))
        if img is None:
            continue
        img = cv2.resize(img, (cell, cell))
        lp  = ldir/(ip.stem+".txt")
        if lp.exists():
            for ln in lp.read_text().strip().splitlines():
                pts = ln.strip().split()
                if len(pts) >= 5:
                    _, cx, cy, bw, bh = map(float, pts[:5])
                    x1 = max(0, int((cx-bw/2)*cell))
                    y1 = max(0, int((cy-bh/2)*cell))
                    x2 = min(cell-1, int((cx+bw/2)*cell))
                    y2 = min(cell-1, int((cy+bh/2)*cell))
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(img, "car", (x1+2, max(y1-4,10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)
        r, c = divmod(i, cols)
        grid[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = img
    path = out/"preview_grid.jpg"
    cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(f"\n  Preview grid → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0  = time.time()
    out = Path(OUTPUT_DIR)
    tmp = out/"_tmp"

    if not ROBOFLOW_API_KEY or ROBOFLOW_API_KEY == "YOUR_API_KEY_HERE":
        print("""
ERROR: No Roboflow API key.
  1. Go to https://app.roboflow.com/settings/api
  2. Copy your key and paste it into ROBOFLOW_API_KEY at the top of this script
""")
        return

    urls = [u.strip() for u in DATASET_URLS if u.strip() and not u.strip().startswith("#")]
    if not urls:
        print("ERROR: DATASET_URLS is empty.")
        return

    print(f"\n{'='*60}")
    print(f"  Roboflow Universe — Full Dataset Builder")
    print(f"  {len(urls)} project(s)  →  {out}/")
    print(f"{'='*60}")

    all_records = []
    sources     = []

    for url in urls:
        parsed = parse_project_url(url)
        if not parsed:
            continue
        workspace, project = parsed
        sources.append(f"universe.roboflow.com/{workspace}/{project}")
        slug = f"{workspace}/{project}"
        print(f"\n── {slug} ──────────────────────────────────────")

        # Method A: ZIP export
        records = try_zip_export(workspace, project, tmp)

        # Method B: CDN fallback
        if not records:
            print(f"  Falling back to CDN download …")
            records = try_cdn_fallback(workspace, project)

        if not records:
            print(f"  [WARN] No images retrieved for {slug} — skipping")
            continue

        print(f"  {slug}: {len(records)} images ready")
        all_records.extend(records)
        print(f"  Running total: {len(all_records)} images")

    if not all_records:
        print("\nERROR: No images downloaded. Check your API key and project URLs.")
        return

    # Shuffle + write
    random.seed(SEED)
    random.shuffle(all_records)
    dirs          = setup_dirs(out)
    counts        = {"train": 0, "val": 0, "test": 0}
    n_boxes_total = 0
    tag_counts    = {}
    total         = len(all_records)

    print(f"\n  Writing {total} images …")
    for idx, (img, boxes, tag) in enumerate(all_records):
        split        = assign_split(idx, total)
        img_d, lbl_d = dirs[split]
        name         = f"f1_{idx:06d}"
        cv2.imwrite(str(img_d/f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_label(lbl_d/f"{name}.txt", boxes)
        counts[split]  += 1
        n_boxes_total  += len(boxes)
        tag_counts[tag] = tag_counts.get(tag, 0) + 1

    write_yaml(out, sources)
    save_preview(out, n=min(total, 32))
    shutil.rmtree(tmp, ignore_errors=True)

    elapsed = time.time() - t0
    report = (
        f"Roboflow Universe Dataset — Build Report\n"
        f"==========================================\n"
        f"Generated  : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Output     : {out.resolve()}\n\n"
        f"Sources\n" +
        "\n".join(f"  {s}" for s in sources) +
        f"\n\nImage counts\n"
        f"  Train / Val / Test : {counts['train']} / {counts['val']} / {counts['test']}\n"
        f"  Total              : {total}\n\n"
        f"Bounding boxes\n"
        f"  Total : {n_boxes_total}\n"
        f"  Avg   : {n_boxes_total / max(total,1):.1f} per image\n\n"
        f"Annotation sources\n" +
        "\n".join(f"  {k:<28} {v} images" for k, v in tag_counts.items()) +
        f"\n\nBuild time : {elapsed:.1f}s\n\n"
        f"Train:\n"
        f"  yolo train model=yolov8s.pt \\\n"
        f"             data={out.resolve()}/data.yaml \\\n"
        f"             epochs=100 batch=16 imgsz={IMG_SIZE}\n"
    )
    (out/"dataset_report.txt").write_text(report)
    print(f"\n{report}")


if __name__ == "__main__":
    main()
