"""Live inference wrapper for our own trained LipNet model.

Uses the exact same mouth-crop preprocessing as training
(MediaPipe 4-keypoint affine alignment -> 96x96 grayscale crop).
"""

import os
import sys

import cv2
import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.join(BASE, "training")
for p in (BASE, TRAINING):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import IDX2CHAR, VOCAB_SIZE, VID_PADDING  # noqa: E402
from model import LipNet, greedy_decode  # noqa: E402
from lipreader_vision import LandmarksDetector  # noqa: E402


class LipNetReader:
    """Lip-reading engine backed by a locally trained LipNet checkpoint."""

    def __init__(self, weights_path: str, device: str = "cuda:0"):
        self.device = device
        self.model = LipNet(in_channels=1, vocab_size=VOCAB_SIZE, dropout_p=0.0).to(device)
        state = torch.load(weights_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.detector = LandmarksDetector()
        self.reference = self._stable_reference()

    def _stable_reference(self) -> np.ndarray:
        ref_path = os.path.join(
            BASE, "lipreader_vision", "mean_face.npy"
        )
        reference = np.load(ref_path)
        return np.vstack(
            [
                np.mean(reference[36:42], axis=0),
                np.mean(reference[42:48], axis=0),
                np.mean(reference[31:36], axis=0),
                np.mean(reference[48:68], axis=0),
            ]
        )

    def detect_frame(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        lm = self.detector.detect(frame_rgb[None, ...], self.detector.short_range_detector)[0]
        if lm is None:
            lm = self.detector.detect(frame_rgb[None, ...], self.detector.full_range_detector)[0]
        return lm

    def crop_frame(self, frame_rgb: np.ndarray, lm: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        transform = cv2.estimateAffinePartial2D(
            lm.astype(np.float32), self.reference.astype(np.float32), method=cv2.LMEDS
        )[0]
        if transform is None:
            return None
        warped = cv2.warpAffine(
            gray, transform, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        mouth = np.matmul(lm, transform[:, :2].T) + transform[:, 2].T  # (4,2)
        cx, cy = mouth[3]
        y0, y1 = int(round(np.clip(cy - 48, 0, 256))), int(round(np.clip(cy + 48, 0, 256)))
        x0, x1 = int(round(np.clip(cx - 48, 0, 256))), int(round(np.clip(cx + 48, 0, 256)))
        crop = warped[y0:y1, x0:x1]
        if crop.shape != (96, 96):
            crop = cv2.resize(crop, (96, 96))
        return crop

    @torch.no_grad()
    def process_sequence(self, frames: list, landmarks: list) -> str:
        crops = []
        for f, lm in zip(frames, landmarks):
            if lm is None:
                continue
            c = self.crop_frame(f, lm)
            if c is not None:
                crops.append(c)
        if len(crops) < 5:
            return ""
        crops = crops[: VID_PADDING]
        T = len(crops)
        seq = np.stack(crops).astype(np.float32) / 255.0
        if T < VID_PADDING:
            seq = np.pad(seq, ((0, VID_PADDING - T), (0, 0), (0, 0)))
        seq = seq[:, None, :, :]  # (T, 1, 96, 96)
        inp = torch.from_numpy(seq).permute(1, 0, 2, 3).unsqueeze(0).to(self.device)  # (1, 1, T, 96, 96)
        logits = self.model(inp)
        preds = greedy_decode(logits, IDX2CHAR)
        return preds[0] if preds else ""

    def process_video_file(self, path: str) -> str:
        cap = cv2.VideoCapture(path)
        frames, landmarks = [], []
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            landmarks.append(self.detect_frame(rgb))
        cap.release()
        return self.process_sequence(frames, landmarks)
