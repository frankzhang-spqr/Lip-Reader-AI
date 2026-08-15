"""Open-vocabulary English lip-reading with the pretrained AutoAVSR model.

AutoAVSR (Pingchuan Ma et al., Imperial College London, Apache 2.0) is a
~250M-parameter ResNet3D + Conformer + Transformer model trained on LRS2 + LRS3 +
VoxCeleb2 + AVSpeech (~3,300 h). It decodes arbitrary English sentences through a
5000-subword SentencePiece vocabulary via joint CTC/attention beam search with an
English RNNLM. On the LRS3 test set it achieves ~19-25% word error rate — the
closest any local model gets to "reading the English dictionary".

Weights (~1 GB) auto-download from Hugging Face on first use (or run
`python download_english_models.py` beforehand).
"""

import os
import sys

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
AUTOAVSR = os.path.join(BASE, "autoavsr")
for p in (BASE, AUTOAVSR):
    if p not in sys.path:
        sys.path.insert(0, p)

import download_english_models  # noqa: E402
from autoavsr.model import AVSR  # noqa: E402
from autoavsr.video_process import VideoProcess  # noqa: E402
from lipreader_vision import LandmarksDetector  # noqa: E402
from lipreader_vision.facemesh import FaceMeshDetector  # noqa: E402

MODEL_DIR = os.path.join(BASE, "models", "LRS3_V_WER19.1")
LM_DIR = os.path.join(BASE, "models", "lm_en_subword")

_NORM_MEAN = 0.421
_NORM_STD = 0.165
_MIN_FRAMES = 25


def video_transform(patches: torch.Tensor, speed_rate: float = 1.0) -> torch.Tensor:
    """Match AutoAVSR's video transform: (T,96,96) uint8 -> (1,T,88,88) float."""
    x = patches.unsqueeze(-1)  # (T, 96, 96, 1)
    if speed_rate != 1.0:
        idx = torch.linspace(
            0, x.shape[0] - 1, int(x.shape[0] / speed_rate), dtype=torch.int64
        )
        x = torch.index_select(x, dim=0, index=idx)
    x = x.permute(3, 0, 1, 2)  # (1, T, 96, 96)
    x = x / 255.0
    x = torchvision.transforms.CenterCrop(88)(x)
    x = F.normalize(x, _NORM_MEAN, _NORM_STD)
    return x


class EnglishReader:
    """Open-vocabulary lip-reading engine backed by pretrained AutoAVSR."""

    def __init__(
        self,
        device: str = "cuda:0",
        detector: str = "facemesh",
        beam_size: int = 40,
        ctc_weight: float = 0.1,
        lm_weight: float = 0.3,
        input_fps: float = 25.0,
    ):
        self.device = device
        self.input_fps = input_fps
        download_english_models.ensure_downloaded()

        model_path = os.path.join(MODEL_DIR, "model.pth")
        model_conf = os.path.join(MODEL_DIR, "model.json")
        rnnlm = os.path.join(LM_DIR, "model.pth")
        rnnlm_conf = os.path.join(LM_DIR, "model.json")

        self.model = AVSR(
            "video",
            model_path,
            model_conf,
            rnnlm,
            rnnlm_conf,
            penalty=0.0,
            ctc_weight=ctc_weight,
            lm_weight=lm_weight,
            beam_size=beam_size,
            device=device,
        )
        self.video_process = VideoProcess(convert_gray=True, mean_face_path="mean_face.npy")

        if detector == "facemesh":
            self._mesh = FaceMeshDetector()
        else:
            self._mesh = None
        self._detector = LandmarksDetector() if self._mesh is None else None

    def detect_frame(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """Return one frame's landmarks: (478, 2) mesh or (4, 2) quad."""
        if self._mesh is not None:
            return self._mesh.detect(frame_rgb)
        lm = self._detector.detect(frame_rgb[None, ...], self._detector.short_range_detector)[0]
        if lm is None:
            lm = self._detector.detect(frame_rgb[None, ...], self._detector.full_range_detector)[0]
        return lm

    def _to_quad(self, lm: np.ndarray) -> np.ndarray:
        if lm.shape[0] > 4:
            return FaceMeshDetector.quad_points(lm)
        return lm

    def process_sequence(self, frames: list, landmarks: list) -> str:
        """Transcribe a recorded lip sequence (open-vocabulary English)."""
        if not frames or not landmarks:
            return ""
        quads = [self._to_quad(lm) if lm is not None else None for lm in landmarks]
        video = np.stack([f.astype(np.uint8) for f in frames], axis=0)
        patches = self.video_process(video, quads)
        if patches is None or len(patches) < _MIN_FRAMES:
            return ""
        data = video_transform(torch.tensor(patches), speed_rate=self.input_fps / 25.0)
        return self.model.infer(data.to(self.device))

    def process_video_file(self, path: str) -> str:
        import cv2

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames, landmarks = [], []
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            landmarks.append(self.detect_frame(rgb))
        cap.release()
        if not frames:
            raise ValueError(f"No frames read from {path}")
        old = self.input_fps
        self.input_fps = fps
        try:
            return self.process_sequence(frames, landmarks)
        finally:
            self.input_fps = old
