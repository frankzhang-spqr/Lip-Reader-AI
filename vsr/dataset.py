"""Generic English lip-reading dataset: cached mouth-crop ROIs + transcripts.

Expects ROI files (T, 96, 96) float32 built by prep.py in config.ROI_DIR and a
split file of lines:  clip_id<TAB>video_path<TAB>text
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

import config


def parse_index(index_path: str) -> list[tuple[str, str, str]]:
    rows = []
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                rows.append((parts[0], parts[1], parts[2]))
    return rows


class CorpusDataset(Dataset):
    def __init__(self, split_path: str, tokenizer, n_frames: int = config.N_FRAMES):
        self.rows = parse_index(split_path)
        self.tokenizer = tokenizer
        self.n_frames = n_frames

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        clip_id, _, text = self.rows[i]
        roi = np.load(os.path.join(config.ROI_DIR, clip_id + ".npy")).astype(np.float32)
        T = roi.shape[0]
        if T > self.n_frames:
            roi = roi[: self.n_frames]
            T = self.n_frames
        padded = np.zeros((self.n_frames, config.CROP_SIZE, config.CROP_SIZE), dtype=np.float32)
        padded[:T] = roi
        video = torch.from_numpy(padded).unsqueeze(1)  # (T, 1, 96, 96)
        labels = torch.tensor(self.tokenizer.text_to_ids(text.lower()), dtype=torch.long)
        return video, T, labels, text


def collate(batch):
    videos, lens, labels, texts = zip(*batch)
    videos = torch.stack(videos).permute(0, 2, 1, 3, 4).contiguous()  # (B, 1, T, 96, 96)
    inp_len = torch.tensor(lens, dtype=torch.long)
    max_len = max(l.size(0) for l in labels)
    padded = torch.zeros(len(labels), max_len, dtype=torch.long)
    tgt_len = torch.zeros(len(labels), dtype=torch.long)
    for i, l in enumerate(labels):
        padded[i, : l.size(0)] = l
        tgt_len[i] = l.size(0)
    return videos, inp_len, padded, tgt_len, texts
