"""
track_mapper/keypoint_detector.py
===================================
ResNet-50 CNN that regresses 2D image coords of stable track landmarks,
then computes a homography H mapping image → real-world track coordinates.

Analogous to court_line_detector/ in abdullahtarek/tennis_analysis.

Landmark examples for Silverstone GP camera 1:
  0  pit lane entry line, left edge
  1  pit lane entry line, right edge
  2  Turn 3 apex curb, inner left
  3  Turn 3 apex curb, inner right
  4  Copse 100m braking board, left post
  5  Copse 100m braking board, right post
  6  Start/finish line, left
  7  Start/finish line, right
"""

from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# ── Circuit definitions ───────────────────────────────────────────────────────

CIRCUIT_KEYPOINTS: dict[str, int] = {
    "silverstone": 8,
    "monaco":      10,
    "monza":       8,
    "spa":         8,
    "default":     8,
}

# Real-world top-down coordinates (metres) for each landmark
# These define the canonical circuit map each homography maps to
CIRCUIT_WORLD_COORDS: dict[str, list[list[float]]] = {
    "silverstone": [
        [  0.0,   0.0],  # 0  pit entry left
        [ 12.0,   0.0],  # 1  pit entry right
        [ 45.0,  80.0],  # 2  T3 apex curb inner left
        [ 57.0,  80.0],  # 3  T3 apex curb inner right
        [120.0, 140.0],  # 4  Copse 100m board left
        [132.0, 140.0],  # 5  Copse 100m board right
        [200.0, 200.0],  # 6  S/F line left
        [212.0, 200.0],  # 7  S/F line right
    ],
}
CIRCUIT_WORLD_COORDS["default"] = CIRCUIT_WORLD_COORDS["silverstone"]


@dataclass
class KeypointResult:
    frame_idx:  int
    keypoints:  np.ndarray   # (N, 2) normalised image coords [0,1]
    homography: np.ndarray   # 3×3 H matrix (image → world)
    valid:      bool = True


class KeypointDetector:
    """
    ResNet-50 regression head predicting N landmark coords per frame.
    Computes homography from predicted landmarks + known world coords.

    Training target: (N*2,) vector of normalised (x, y) landmark coords.
    Loss: MSE over normalised coords.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]
    INPUT_SIZE    = 224

    def __init__(
        self,
        model_path: str | None = None,
        circuit:    str = "default",
        device:     str | None = None,
    ):
        self.circuit     = circuit
        self.n_kp        = CIRCUIT_KEYPOINTS.get(circuit, 8)
        self.world_coords = np.array(
            CIRCUIT_WORLD_COORDS.get(circuit, CIRCUIT_WORLD_COORDS["default"]),
            dtype=np.float32,
        )
        self._model      = None
        self._model_path = model_path
        self._device_str = device
        self._transform  = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            import torchvision.transforms as T
            from torchvision import models
            import torch.nn as nn
        except ImportError:
            raise ImportError("pip install torch torchvision")

        import torch
        self._device = torch.device(
            self._device_str or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Build ResNet-50 with regression head
        m     = models.resnet50(weights="IMAGENET1K_V2")
        m.fc  = nn.Linear(m.fc.in_features, self.n_kp * 2)
        if self._model_path and Path(self._model_path).exists():
            state = torch.load(self._model_path, map_location=self._device)
            m.load_state_dict(state)
        m.to(self._device).eval()
        self._model = m

        import torchvision.transforms as T
        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> KeypointResult:
        """Detect keypoints in a single BGR frame."""
        self._load()
        import torch, cv2
        H, W = frame.shape[:2]
        inp  = self._transform(frame).unsqueeze(0).to(self._device)
        with torch.no_grad():
            pred = self._model(inp).cpu().numpy().reshape(-1, 2)  # (N, 2) normalised
        kp_pixels = pred * np.array([W, H], dtype=np.float32)
        hom       = self._compute_homography(kp_pixels)
        return KeypointResult(
            frame_idx  = frame_idx,
            keypoints  = pred,
            homography = hom if hom is not None else np.eye(3, dtype=np.float32),
            valid      = hom is not None,
        )

    def detect_batch(
        self,
        frames: list[np.ndarray],
        stride: int = 30,
    ) -> dict[int, KeypointResult]:
        """Detect every `stride` frames. Returns {frame_idx: KeypointResult}."""
        results = {}
        for i in range(0, len(frames), stride):
            results[i] = self.detect(frames[i], frame_idx=i)
        return results

    def get_homography_at(
        self,
        frame_idx:      int,
        keypoint_frames: dict[int, KeypointResult],
    ) -> np.ndarray:
        """Return the nearest valid homography for a given frame index."""
        if frame_idx in keypoint_frames and keypoint_frames[frame_idx].valid:
            return keypoint_frames[frame_idx].homography
        candidates = [
            (abs(fi - frame_idx), fi)
            for fi, r in keypoint_frames.items()
            if r.valid
        ]
        if not candidates:
            return np.eye(3, dtype=np.float32)
        _, nearest = min(candidates)
        return keypoint_frames[nearest].homography

    # ── Training helper ───────────────────────────────────────────────────────

    @staticmethod
    def train(
        annotations_dir: str,
        circuit:         str   = "default",
        epochs:          int   = 50,
        lr:              float = 1e-4,
        save_path:       str   = "models/keypoint_cnn.pth",
    ) -> None:
        """
        Fine-tune the ResNet-50 keypoint regressor.
        Annotation JSON format:
          {"image": "frame.jpg", "keypoints": [[x1,y1], [x2,y2], ...]}
        """
        try:
            import torch, torch.nn as nn, torch.optim as optim
            import torchvision.transforms as T
            from torchvision import models
            from torch.utils.data import DataLoader, Dataset
            from PIL import Image as PILImage
        except ImportError:
            raise ImportError("pip install torch torchvision")

        n_kp   = CIRCUIT_KEYPOINTS.get(circuit, 8)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ann_dir = Path(annotations_dir)

        records = []
        for jf in sorted(ann_dir.glob("*.json")):
            d        = json.loads(jf.read_text())
            img_path = ann_dir.parent / "images" / d["image"]
            kps      = np.array(d["keypoints"], dtype=np.float32)
            records.append((str(img_path), kps))

        print(f"  Training keypoint CNN: {len(records)} annotated frames, {n_kp} landmarks")

        class KPDataset(Dataset):
            def __init__(self, records):
                self.records = records
                self.tf = T.Compose([
                    T.Resize((224, 224)),
                    T.ColorJitter(brightness=0.2, contrast=0.2),
                    T.ToTensor(),
                    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
                ])
            def __len__(self): return len(self.records)
            def __getitem__(self, i):
                img_path, kps = self.records[i]
                img = PILImage.open(img_path).convert("RGB")
                w, h = img.size
                kps_norm = kps / np.array([w, h], dtype=np.float32)
                return self.tf(img), torch.tensor(kps_norm.flatten(), dtype=torch.float32)

        m    = models.resnet50(weights="IMAGENET1K_V2")
        m.fc = nn.Linear(m.fc.in_features, n_kp * 2)
        m.to(device)
        opt    = optim.Adam(m.parameters(), lr=lr)
        loss_fn= nn.MSELoss()
        loader = DataLoader(KPDataset(records), batch_size=16, shuffle=True)

        m.train()
        for ep in range(epochs):
            total = 0.0
            for imgs, targets in loader:
                imgs, targets = imgs.to(device), targets.to(device)
                loss = loss_fn(m(imgs), targets)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
            print(f"    Epoch {ep+1}/{epochs}  loss={total/len(loader):.6f}")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(m.state_dict(), save_path)
        print(f"  Saved keypoint model → {save_path}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute_homography(self, kp_pixels: np.ndarray) -> np.ndarray | None:
        try:
            import cv2
        except ImportError:
            return np.eye(3, dtype=np.float32)
        if len(kp_pixels) < 4:
            return None
        src = kp_pixels.astype(np.float32)
        dst = self.world_coords.astype(np.float32)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return H.astype(np.float32) if H is not None else None
