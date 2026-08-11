"""Training configuration and vocabulary."""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT, "raw")
VIDEO_EXTRACT_DIR = os.path.join(RAW_DIR, "video_extracted")
ALIGN_EXTRACT_DIR = os.path.join(RAW_DIR, "align_extracted")
CACHE_DIR = os.path.join(ROOT, "cache")
DATA_DIR = os.path.join(ROOT, "data")
WEIGHTS_DIR = os.path.join(ROOT, "weights")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# GRID sentence vocabulary ordering for WER (word ordering)
GRID_WORDS = [
    "bin", "lay", "place", "set", "put",
    "blue", "green", "red", "white",
    "at", "by", "in", "with", "and",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "x", "y", "z",
    "now", "please", "soon",
]
GRID_WORDS = list(dict.fromkeys(GRID_WORDS))  # dedupe, keep order

# CTC vocabulary: letters a-z, space, blank
CHARS = [chr(i) for i in range(ord("a"), ord("z") + 1)] + [" "]
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
BLANK_IDX = len(CHARS)  # 27
VOCAB_SIZE = len(CHARS) + 1
IDX2CHAR = CHARS + ["<blank>"]

# GRID speakers
TEST_SPEAKERS = [1, 2, 20, 22]
TRAIN_SPEAKERS = [s for s in range(1, 35) if s not in TEST_SPEAKERS and s != 21]

# Hyperparameters
VID_PADDING = 75  # frames
TXT_PADDING = 200  # chars
BATCH_SIZE = 16
BASE_LR = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0
MAX_EPOCH = 300
DISPLAY_ITERS = 50
SAVE_EVERY_ITERS = 1000
CROP_SIZE = 96


def text_to_idx(text: str) -> list[int]:
    """Map a lower-case sentence to a CTC char index list."""
    return [CHAR2IDX[c] for c in text.lower()]


def idx_to_text(idx: list[int]) -> str:
    return "".join(IDX2CHAR[i] for i in idx)
