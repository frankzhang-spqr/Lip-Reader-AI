"""GRID dataset: build mouth-crop cache and provide a PyTorch Dataset.

The mouth crops use the same MediaPipe 4-keypoint + affine pipeline as live
inference (see lipreader_vision/detector.py), so the model sees identical input
in training and in the live webcam app.
"""

import os
import re
import sys
import threading

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import CACHE_DIR, CHAR2IDX, CROP_SIZE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (BASE,):
    if p not in sys.path:
        sys.path.insert(0, p)
from lipreader_vision import LandmarksDetector  # noqa: E402

_thread_local = threading.local()


def parse_index(index_path: str) -> list[tuple[str, str, str, str]]:
    """Read index file lines: clip_id \\t video \\t align \\t label."""
    rows = []
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 4:
                rows.append(tuple(parts))
    return rows


def _stable_reference():
    ref_path = os.path.join(BASE, "lipreader_vision", "mean_face.npy")
    reference = np.load(ref_path)
    return np.vstack(
        [
            np.mean(reference[36:42], axis=0),
            np.mean(reference[42:48], axis=0),
            np.mean(reference[31:36], axis=0),
            np.mean(reference[48:68], axis=0),
        ]
    )


def _detect_landmarks(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    detector = getattr(_thread_local, "detector", None)
    if detector is None:
        detector = LandmarksDetector()
        _thread_local.detector = detector
    lm = None
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % 10 == 0:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            lm = detector.detect(rgb[None, ...], detector.short_range_detector)[0]
            if lm is None:
                lm = detector.detect(rgb[None, ...], detector.full_range_detector)[0]
            if lm is not None:
                break
        idx += 1
    cap.release()
    return lm


def crop_clip(video_path: str, lm: np.ndarray, stable_ref: np.ndarray) -> np.ndarray | None:
    """Read an MPEG clip and return mouth-crop frames (T, 96, 96) uint8."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    transform = cv2.estimateAffinePartial2D(
        lm.astype(np.float32), stable_ref.astype(np.float32), method=cv2.LMEDS
    )[0]
    if transform is None:
        cap.release()
        return None
    tl = np.matmul(lm, transform[:, :2].T) + transform[:, 2].T  # (4, 2)
    cx, cy = tl[3]
    half = CROP_SIZE // 2
    out = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        warped = cv2.warpAffine(
            gray, transform, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        y0, y1 = int(round(np.clip(cy - half, 0, 256))), int(round(np.clip(cy + half, 0, 256)))
        x0, x1 = int(round(np.clip(cx - half, 0, 256))), int(round(np.clip(cx + half, 0, 256)))
        crop = warped[y0:y1, x0:x1]
        if crop.shape != (CROP_SIZE, CROP_SIZE):
            crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
        out.append(crop)
    cap.release()
    if not out:
        return None
    return np.stack(out).astype(np.uint8)


def _cache_worker(row):
    """Module-level worker for ProcessPoolExecutor."""
    clip_id, video, align, label = row
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", clip_id)
    out_path = os.path.join(CACHE_DIR, f"{safe}.npy")
    if os.path.isfile(out_path):
        return (clip_id, True)
    lm = _detect_landmarks(video)
    if lm is None:
        return (clip_id, False)
    seq = crop_clip(video, lm, _stable_reference())
    if seq is None:
        return (clip_id, False)
    np.save(out_path, seq)
    return (clip_id, True)


def build_cache(index_path: str, cache_dir: str = CACHE_DIR, num_workers: int = 8) -> None:
    """Build .npy mouth-crop caches for every clip in the index file.

    Uses threads: MediaPipe detection and cv2 video decoding release the GIL,
    so threads give near-process parallelism without Windows spawn issues.
    """
    from concurrent.futures import ThreadPoolExecutor

    rows = parse_index(index_path)
    os.makedirs(cache_dir, exist_ok=True)

    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for clip_id, success in pool.map(_cache_worker, rows):
            if success:
                ok += 1
            else:
                fail += 1
                print(f"  failed: {clip_id}")
    print(f"cache built: ok={ok} fail={fail} of {len(rows)}")


class GRIDDataset(Dataset):
    """Loads cached mouth-crop clips (T,96,96) uint8 plus their char label."""

    def __init__(self, index_path: str, cache_dir: str = CACHE_DIR, vid_padding: int = 75):
        self.rows = parse_index(index_path)
        self.cache_dir = cache_dir
        self.vid_padding = vid_padding

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        clip_id, video, align, label = self.rows[i]
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", clip_id)
        seq = np.load(os.path.join(self.cache_dir, f"{safe}.npy"))
        if seq.shape[0] > self.vid_padding:
            seq = seq[: self.vid_padding]
        T = seq.shape[0]
        pad = self.vid_padding - T
        if pad > 0:
            seq = np.pad(seq, ((0, pad), (0, 0), (0, 0)))
        seq = seq.astype(np.float32) / 255.0
        seq = seq[:, None, :, :]  # (T, 1, 96, 96)
        label_idx = [CHAR2IDX[c] for c in label.lower() if c in CHAR2IDX]
        return torch.from_numpy(seq), torch.tensor(label_idx, dtype=torch.long), T, label
