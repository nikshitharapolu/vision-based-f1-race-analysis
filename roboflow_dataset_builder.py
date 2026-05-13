"""
Roboflow Dataset Builder 
==================================================
Roboflow Universe PROJECT URL:
    

The script downloads the ENTIRE dataset from each project URL automatically:
  1. Hits the Roboflow API to find the project + latest version number
  2. Requests a YOLOv8 export ZIP for that version (images + labels bundled)
  3. Downloads and extracts the ZIP — already has bounding-box .txt files
  4. If no export ZIP is available → falls back to search_all() + source CDN
     download + YOLOv8 auto-label
  5. Merges multiple projects into one unified dataset with a shared CLASS_MAP
  6. Writes a clean data.yaml + preview_grid.jpg for QA

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
from config import *


ROBOFLOW_API_KEY = ROBOFLOW_API_KEY
# Free at: https://app.roboflow.com/settings/api

DATASET_URLS = [
    "https://universe.roboflow.com/aio-project/formula-1-dkxin",
    "https://universe.roboflow.com/project-ops31/formula-1-sexm0",
    "https://universe.roboflow.com/shreyas-workspace-8fcpm/datasetprep-dunei",
    "https://universe.roboflow.com/cys-space/f1tracksegmentation-iheeq",
    "https://universe.roboflow.com/f1-vision/f1-vision-lxwu4",
    "https://universe.roboflow.com/nitin-fatsh/formula-btv10",
    "https://universe.roboflow.com/nikshithas-workspace/f1-race-track-limits",
    "https://universe.roboflow.com/yoav-fogel-yia3f/f1-car-recognition",
    "https://universe.roboflow.com/jayanths-workspace/formula-one-car-detection",
]


OUTPUT_DIR      = "roboflow_dataset"
EXPORT_FORMAT   = "yolov8"
IMG_SIZE        = 640
CONF_THRESHOLD  = 0.25
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15
SEED            = 42
REQUEST_TIMEOUT = 60
MAX_IMAGES      = None  
RF_API        = "https://api.roboflow.com"
RF_SOURCE_CDN = "https://source.roboflow.com"


SEGMENTATION_PROJECTS = {
    "f1tracksegmentation-iheeq",  
}

def export_format_for(project: str) -> str:
    """Return the correct Roboflow export format string for this project."""
    return "yolov8-seg" if project in SEGMENTATION_PROJECTS else "yolov8"


LABEL_REMAP: dict[str, str] = {
    # Generic car
    "Non-penalty":                   "__drop__",
    "f1car - v3 2024-05-13 1-30pm":  "car",
    "f1car":                         "car",
    # Red Bull
    "RedBull":                       "RedBull",
    "redbull":                       "RedBull",
    "Red Bull":                      "RedBull",
    "red bull":                      "RedBull",
    "Red-Bull-Racing":               "RedBull",
    # Mercedes
    "Mercedes":                      "Mercedes",
    "mercedes":                      "Mercedes",
    # Ferrari
    "Ferrari":                       "Ferrari",
    "ferrari":                       "Ferrari",
    # McLaren
    "McLaren":                       "McLaren",
    "Mclaren":                       "McLaren",
    "mclaren":                       "McLaren",
    # Alpine
    "Alpine":                        "Alpine",
    "alpine":                        "Alpine",
    # Aston Martin
    "AstonMartin":                   "AstonMartin",
    "Aston-Martin":                   "AstonMartin",
    "aston martin":                  "AstonMartin",
    "aston":                         "AstonMartin",
    "Aston Martin":                  "AstonMartin",
    # Williams (incl. typo)
    "Williams":                      "Williams",
    "williams":                      "Williams",
    "wiliams":                       "Williams",
    # Haas
    "haas":                          "Haas",
    "Haas":                          "Haas",
    # Kick Sauber (was Alfa Romeo)
    "kick sauber":                   "KickSauber",
    "Kick Sauber":                   "KickSauber",
    "alfa":                          "KickSauber",
    "Alfa":                          "KickSauber",
    # Racing Bulls (was AlphaTauri)
    "Racing Bulls":                  "RacingBulls",
    "racing bulls":                  "RacingBulls",
    "alpha":                         "RacingBulls",
    "Alpha":                         "RacingBulls",
    # Track surface 
    "track":                         "track_surface",
    "track_surface":                 "track_surface",
    # Track limit labels 
    "inside":                        "on_track",
    "outside":                       "off_track",
    "track_inside":                  "on_track",
    "track_outside":                 "off_track",
    "out of track":                  "off_track",
    "off_track":                     "off_track",

    # Race incidents & events
    "crash":                         "crash",
    "Crash":                         "crash",
    "Penalty":                       "penalty_car",
    "pitstop":                       "pitstop",
    "Pitstop":                       "pitstop",
    "race start":                    "race_start",
    "Race Start":                    "race_start",
    "marshals":                      "marshal",
    "Marshals":                      "marshal",
    "yellow flag situation":         "yellow_flag",
    "Yellow Flag Situation":         "yellow_flag",
    # Dropped labels — no bounding-box
    "maar":                          "__drop__",
    "apex":                          "__drop__",
    "clean racing":                  "__drop__",
    "podium":                        "__drop__",
    "Clean Racing":                  "__drop__",
    "Podium":                        "__drop__",
    "Alfa-Romeo":                    "__drop__",
    "Alpha-Tauri":                   "__drop__",
}

UNIFIED_CLASSES: list[str] = [
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
    "off_track",
    "on_track",
]

CLASS_MAP: dict[str, int] = {name: idx for idx, name in enumerate(UNIFIED_CLASSES)}

_UNKNOWN_LABELS: dict[str, int] = {}  


def remap_label(raw_label: str) -> str:
    """
    Normalise a raw dataset label to a unified class name.
    Returns "__drop__" for labels that should be skipped entirely.
    Falls back to "car" for any unknown label, but logs the miss
    so it appears in the final unknown-labels report.
    Tries exact match first, then case-insensitive match.
    """
    s = raw_label.strip()
    if s in LABEL_REMAP:
        return LABEL_REMAP[s]
    s_lower = s.lower()
    for k, v in LABEL_REMAP.items():
        if k.lower() == s_lower:
            return v
    _UNKNOWN_LABELS[s] = _UNKNOWN_LABELS.get(s, 0) + 1
    return "car"


def print_unknown_labels_report() -> None:
    """
    Print a report of every label seen at runtime that was not in LABEL_REMAP.
    """
    SEP = "-" * 60
    if not _UNKNOWN_LABELS:
        print("  No unknown labels — all labels were mapped correctly.")
        return
    print("\n" + SEP)
    print(f"  WARNING: {len(_UNKNOWN_LABELS)} UNKNOWN LABEL(S) FOUND")
    print("  These were NOT in LABEL_REMAP and silently mapped to 'car'.")
    print("  Add them to LABEL_REMAP with the correct target class.\n")
    print(f"  {'Raw label':<40} {'Count':>6}")
    print("  " + "-" * 48)
    for label, count in sorted(_UNKNOWN_LABELS.items(), key=lambda x: -x[1]):
        print(f"  {repr(label):<40} {count:>6}")
    print(SEP)
    print("\n  To fix: add each line below to LABEL_REMAP in the script:")
    for label in list(_UNKNOWN_LABELS.keys())[:5]:
        print(f'    "{label}": "car",  # TODO: set correct class')
    print()


def get_class_id(raw_label: str):
    """
    Map a raw label string to its unified integer class ID.
    Returns None if the label should be dropped (not written to label file).
    """
    unified = remap_label(raw_label)
    if unified == "__drop__":
        return None
    return CLASS_MAP.get(unified, 0)


def get_or_create_class_id(label: str) -> int:
    unified = remap_label(label)
    if unified == "__drop__":
        return 0
    return CLASS_MAP.get(unified, 0)

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



def load_class_names_from_yaml(extract_dir: Path) -> dict[int, str]:
    """
    Read data.yaml from a Roboflow export ZIP and return {old_class_id: label_name}.
    Falls back to {0: "car"} if yaml is missing or unreadable.
    Prints discovered labels so you can see exactly what the dataset contains.
    """
    yaml_files = list(extract_dir.rglob("data.yaml"))
    if not yaml_files:
        print("    [WARN] No data.yaml found in ZIP — defaulting to {0: 'car'}")
        return {0: "car"}
    try:
        import yaml
        data  = yaml.safe_load(yaml_files[0].read_text())
        names = data.get("names", {})
        if isinstance(names, list):
            result = {i: n for i, n in enumerate(names)}
        elif isinstance(names, dict):
            result = {int(k): v for k, v in names.items()}
        else:
            result = {0: "car"}
        print(f"    Labels in this dataset ({len(result)}):")
        for cid, name in sorted(result.items()):
            mapped = remap_label(name)
            status = "→ FALLBACK 'car' (add to LABEL_REMAP!)" if (mapped == "car" and name.lower() not in ["car","f1car"]) else f"→ {mapped}"
            print(f"      [{cid}] {name!r:<35} {status}")
        return result
    except Exception as e:
        print(f"  [WARN] Could not read data.yaml: {e}")
    return {0: "car"}


def parse_project_url(url: str):
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


def polygon_to_bbox(coords: list[float]) -> tuple[float, float, float, float] | None:
    """
    Convert a flat list of normalised polygon points [x1,y1,x2,y2,...] to a YOLO bounding box (cx, cy, w, h).
    Returns None if the polygon is degenerate (< 3 points or zero area).
    """
    if len(coords) < 6:  
        return None
    try:
        pts  = [(coords[i], coords[i+1]) for i in range(0, len(coords)-1, 2)]
        xs   = [p[0] for p in pts]
        ys   = [p[1] for p in pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        w = x_max - x_min
        h = y_max - y_min
        if w < 0.001 or h < 0.001:
            return None
        return (
            max(0.0, min(1.0, (x_min + x_max) / 2)),
            max(0.0, min(1.0, (y_min + y_max) / 2)),
            max(0.001, min(1.0, w)),
            max(0.001, min(1.0, h)),
        )
    except Exception:
        return None


def segmentation_points_to_bbox(
    points: list[dict], img_w: float, img_h: float
) -> tuple[float, float, float, float] | None:
    """
    Convert Roboflow API segmentation points list
    [{"x": px, "y": py}, ...] to YOLO bbox (cx, cy, w, h).
    Points are in pixel coords; normalised by img_w / img_h.
    """
    try:
        xs = [float(p["x"]) / img_w for p in points]
        ys = [float(p["y"]) / img_h for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        w = x_max - x_min
        h = y_max - y_min
        if w < 0.001 or h < 0.001:
            return None
        return (
            max(0.0, min(1.0, (x_min + x_max) / 2)),
            max(0.0, min(1.0, (y_min + y_max) / 2)),
            max(0.001, min(1.0, w)),
            max(0.001, min(1.0, h)),
        )
    except Exception:
        return None



def get_project_type(workspace: str, project: str) -> str:
    """
    Return the Roboflow project type string:
      "object-detection" | "instance-segmentation" | "semantic-segmentation"
      | "classification" | "unknown"
    """
    data = api_get(f"/{workspace}/{project}")
    if not data:
        return "unknown"
    proj = data.get("project", {})
    return proj.get("type", "unknown")


def get_latest_version(workspace: str, project: str) -> tuple[int | None, str]:
    """
    Return (latest_version_number, project_type).
    version_number is None if no versions exist.
    project_type is one of: object-detection, instance-segmentation,
      semantic-segmentation, classification, unknown.
    """
    data = api_get(f"/{workspace}/{project}")
    if not data:
        return None, "unknown"

    proj      = data.get("project", {})
    proj_type = proj.get("type", "unknown")
    versions = data.get("versions", [])
    if not isinstance(versions, list) or not versions:
        print(f"    Project type: {proj_type}  |  No versions published yet")
        return None, proj_type

    best = None
    for v in versions:
        vid = v.get("id", "")
        m   = re.search(r"/(\d+)$", str(vid))
        if m:
            n = int(m.group(1))
            if best is None or n > best:
                best = n

    print(f"    Project type: {proj_type}  |  Latest version: {best}")
    return best, proj_type


def get_export_zip_url(workspace: str, project: str, version: int) -> str | None:
    """
    Request an export download link. Tries multiple formats in priority order:
      1. Project-specific format (yolov8-seg for known segmentation projects)
      2. yolov8 (standard detection)
    Returns the export.link URL, or None if nothing is available.
    """
    formats_to_try = []

    primary = export_format_for(project)
    formats_to_try.append(primary)
    for fallback in ["yolov8", "yolov8-seg", "coco"]:
        if fallback not in formats_to_try:
            formats_to_try.append(fallback)

    for fmt in formats_to_try:
        data = api_get(f"/{workspace}/{project}/{version}/{fmt}")
        if not data:
            continue
        link = data.get("export", {}).get("link")
        if link:
            if fmt != primary:
                print(f"    Note: using {fmt} export (primary {primary} had no link)")
            return link

    print(f"    No export link available for any format — project may need a version generated on Roboflow first")
    return None


def collect_from_zip(extract_dir: Path) -> list[tuple[Path, Path | None]]:
    
    pairs    = []
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for img_path in sorted(extract_dir.rglob("*")):
        if img_path.suffix.lower() not in img_exts:
            continue
        if "preview" in img_path.name.lower():
            continue
        lbl_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
        if not lbl_path.exists():
            lbl_path = img_path.with_suffix(".txt")
        pairs.append((img_path, lbl_path if lbl_path.exists() else None))
    return pairs


def load_yolo_label(lbl_path: Path, class_id_to_name: dict[int, str]) -> list[tuple]:
    """
    Read a YOLO .txt label file and remap class IDs into the global CLASS_MAP.
    Returns list of (unified_cls_id, cx, cy, w, h).
    """
    boxes = []
    for line in lbl_path.read_text().strip().splitlines():
        pts = line.strip().split()
        if len(pts) < 5:
            continue
        try:
            old_cls    = int(pts[0])
            coords     = list(map(float, pts[1:]))
            label      = class_id_to_name.get(old_cls, f"class_{old_cls}")
            unified_id = get_class_id(label)
            if unified_id is None:
                continue   

            n = len(coords)
            if n == 4:
                cx, cy, w, h = coords
            elif n >= 6 and n % 2 == 0:
                result = polygon_to_bbox(coords)
                if result is None:
                    continue
                cx, cy, w, h = result
            else:
                continue   

            if w > 0.001 and h > 0.001:
                boxes.append((unified_id, cx, cy, w, h))
        except (ValueError, IndexError):
            continue
    return boxes


def try_zip_export(workspace: str, project: str, tmp: Path) -> list | None:
    """
    Download the full YOLOv8 export ZIP for the latest version.
    Returns list of (bgr_img, yolo_boxes, tag) or None if unavailable.
    """
    version, proj_type = get_latest_version(workspace, project)
    if version is None:
        print(f"    No versions found for {workspace}/{project}")
        return None

    if proj_type == "classification":
        print(f"    Skipping: project type is 'classification' (image-level labels only, no bounding boxes)")
        return None

    print(f"    Requesting export (version {version}, type: {proj_type}) …")
    zip_url = get_export_zip_url(workspace, project, version)
    if not zip_url:
        print(f"    No export URL for version {version} — will try CDN fallback")
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
        print(f"    Bad ZIP: {e}")
        return None

    class_id_to_name = load_class_names_from_yaml(zip_dir)
    print(f"    Classes in this dataset: {class_id_to_name}")

    pairs = collect_from_zip(zip_dir)
    print(f"    ZIP contains {len(pairs)} images")
    if not pairs:
        return None

    records = []
    for img_path, lbl_path in pairs:
        if MAX_IMAGES and len(records) >= MAX_IMAGES:
            break

        img = cv2.imread(str(img_path))
        if img is None:
            img = bytes_to_bgr(img_path.read_bytes())
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        if lbl_path and lbl_path.exists():
            boxes = load_yolo_label(lbl_path, class_id_to_name)
            tag   = "zip_yolo_labels"
        else:
            raw   = auto_label(img)
            boxes = [(get_or_create_class_id("car"), cx, cy, bw, bh)
                     for cx, cy, bw, bh in raw]
            tag   = "yolo_autolabel"

        records.append((img, boxes, tag))

    return records if records else None


#  search_all + source CDN  (fallback)

def search_all_images(workspace: str, project: str) -> list[dict]:
    #Page through the search API and return all image records
    records = []
    offset  = 0
    limit   = 200

    print(f"    Fetching image list via search API …")
    while True:
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
            data = api_get(
                f"/{workspace}/{project}/search",
                {"offset": offset, "limit": limit,
                 "in_dataset": "true", "fields": "id,name,owner,annotations"},
            )
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
    #Download one image from source CDN + fetch its annotations. Returns (img, boxes)
    owner    = record.get("owner", workspace)
    image_id = record["id"]

    img_data = download_bytes(
        f"{RF_SOURCE_CDN}/{owner}/{image_id}/original.jpg", image_id[:16]
    )
    if not img_data:
        return None, []

    img = bytes_to_bgr(img_data)
    if img is None:
        return None, []
    boxes = []
    det   = api_get(f"/{workspace}/{project}/images/{image_id}")
    if det:
        img_info   = det.get("image", {})
        annotation = img_info.get("annotation") or {}
        raw_boxes  = annotation.get("boxes", [])
        try:
            img_w = float(annotation.get("width",  img.shape[1]))
            img_h = float(annotation.get("height", img.shape[0]))
        except (TypeError, ValueError):
            img_h, img_w = img.shape[:2]

        for b in raw_boxes:
            label      = b.get("label", "car")
            unified_id = get_class_id(label)
            if unified_id is None:
                continue  
            points = b.get("points")
            if points and isinstance(points, list) and len(points) >= 3:
                result = segmentation_points_to_bbox(points, img_w, img_h)
                if result is None:
                    continue
                cx, cy, bw, bh = result
            else:
                # Object detection: API returns x, y, width, height (pixel, centred)
                try:
                    x  = float(b.get("x", 0))
                    y  = float(b.get("y", 0))
                    bw = float(b.get("width",  0))
                    bh = float(b.get("height", 0))
                except (TypeError, ValueError):
                    continue
                cx = x / img_w;  cy = y / img_h
                bw = bw / img_w; bh = bh / img_h

            if bw > 0.001 and bh > 0.001:
                boxes.append((
                    unified_id,
                    max(0., min(1., cx)), max(0., min(1., cy)),
                    max(0.001, min(1., bw)), max(0.001, min(1., bh)),
                ))

    return img, boxes


def try_cdn_fallback(workspace: str, project: str) -> list:
    #Method B: search_all + CDN download + annotation API per image
    proj_type = get_project_type(workspace, project)
    if proj_type == "classification":
        print(f"    Skipping CDN fallback: project type is 'classification' (no bounding boxes)")
        return []

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
            raw   = auto_label(img)
            boxes = [(get_or_create_class_id("car"), cx, cy, bw, bh)
                     for cx, cy, bw, bh in raw]
            tag   = "yolo_autolabel"
        else:
            tag = "api_annotations"
        records.append((img, boxes, tag))
        if (i+1) % 25 == 0 or (i+1) == total:
            print(f"    … {i+1}/{total} images downloaded")

    return records



_yolo_model = None

def auto_label(image: np.ndarray) -> list[tuple]:
    #Run YOLOv8n on image; returns 4-tuples (cx, cy, bw, bh). Callers add class_id
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            print("\n  Loading YOLOv8n for auto-label fallback …")
            _yolo_model = YOLO("yolov8n.pt")
        except ImportError:
            print("  [WARN] pip install ultralytics  for auto-label")
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
                max(0., min(1., ((x1+x2)/2) / W)),
                max(0., min(1., ((y1+y2)/2) / H)),
                max(0.001, min(1., (x2-x1) / W)),
                max(0.001, min(1., (y2-y1) / H)),
            ))
    return boxes



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


def write_label(path: Path, boxes: list[tuple]) -> None:
    path.write_text(
        "\n".join(
            f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            for cls, cx, cy, bw, bh in boxes
        )
    )


def write_yaml(out: Path, sources: list[str]) -> None:
    src_lines   = "\n".join(f"#   {s}" for s in sources)
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(UNIFIED_CLASSES))

    (out/"data.yaml").write_text(f"""\
# F1 Car Detection — Roboflow Universe Dataset
# Sources:
{src_lines}
#
# Class schema:
#   0  car          — generic car on track  (Dataset 1: Non-penalty + unknown labels)
#   1  penalty_car  — car under a penalty   (Dataset 1: Penalty)
#   2  RedBull      — Red Bull car          (Dataset 2)
#   3  Alpine       — Alpine car            (Dataset 2)
#   4  McLaren      — McLaren car           (Dataset 2)
#   5  Mercedes     — Mercedes car          (Dataset 2)
#   6  AstonMartin  — Aston Martin car      (Dataset 2)
#   7  Ferrari      — Ferrari car           (Dataset 2)
#   8  Williams     — Williams car          (Dataset 2)

path: {out.resolve()}
train: train/images
val:   val/images
test:  test/images

nc: {len(UNIFIED_CLASSES)}
names:
{names_block}
""")


def save_preview(out: Path, n: int = 16, cols: int = 4, cell: int = 160) -> None:
    idir   = out/"train"/"images"
    ldir   = out/"train"/"labels"
    images = sorted(idir.glob("*.jpg"))[:n]
    if not images:
        return
    id_to_name = {cid: name for name, cid in CLASS_MAP.items()}
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
                    cls_id = int(pts[0])
                    cx, cy, bw, bh = map(float, pts[1:5])
                    x1 = max(0, int((cx-bw/2)*cell))
                    y1 = max(0, int((cy-bh/2)*cell))
                    x2 = min(cell-1, int((cx+bw/2)*cell))
                    y2 = min(cell-1, int((cy+bh/2)*cell))
                    label = id_to_name.get(cls_id, str(cls_id))
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(img, label, (x1+2, max(y1-4,10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)
        r, c = divmod(i, cols)
        grid[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = img

    path = out/"preview_grid.jpg"
    cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(f"\n  Preview grid → {path}")

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
        print(f"\n── {slug} ────────────────────────────────────────")
        records = try_zip_export(workspace, project, tmp)
        if not records:
            print(f"  ZIP not available — falling back to CDN download …")
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
    random.seed(SEED)
    random.shuffle(all_records)
    dirs          = setup_dirs(out)
    counts        = {"train": 0, "val": 0, "test": 0}
    n_boxes_total = 0
    tag_counts: dict[str, int] = {}
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
        "Roboflow Universe Dataset — Build Report\n"
        "==========================================\n"
        f"Generated  : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Output     : {out.resolve()}\n\n"
        "Sources\n" +
        "\n".join(f"  {s}" for s in sources) +
        "\n\nUnified class schema\n" +
        "\n".join(f"  {idx}: {name}" for idx, name in enumerate(UNIFIED_CLASSES)) +
        f"\n\nImage counts\n"
        f"  Train / Val / Test : {counts['train']} / {counts['val']} / {counts['test']}\n"
        f"  Total              : {total}\n\n"
        f"Bounding boxes\n"
        f"  Total : {n_boxes_total}\n"
        f"  Avg   : {n_boxes_total / max(total, 1):.1f} per image\n\n"
        "Annotation sources\n" +
        "\n".join(f"  {k:<30} {v} images" for k, v in tag_counts.items()) +
        f"\n\nBuild time : {elapsed:.1f}s\n\n"
        "Train:\n"
        "  yolo train model=yolov8s.pt \\\n"
        f"             data={out.resolve()}/data.yaml \\\n"
        f"             epochs=100 batch=16 imgsz={IMG_SIZE}\n"
    )
    (out/"dataset_report.txt").write_text(report)
    print(f"\n{report}")
    print_unknown_labels_report()


if __name__ == "__main__":
    main()