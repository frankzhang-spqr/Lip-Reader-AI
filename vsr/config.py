"""Conformer VSR training config (open-vocabulary, SentencePiece subwords).

This is the "train your own big model" path. Feed it any English lip-reading
corpus (see DATA.md for how to obtain LRS3 / VoxCeleb2) in the format:

    corpus/<split>.txt      clip_id<TAB>video_path<TAB>text   (one per line)
    corpus/raw/<clip_id>.mp4                                  (or .avi/.mkv)

Then run prep.py to build mouth-crop ROIs and train.py to train.
"""

import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CORPUS_DIR = os.path.join(BASE, "corpus")   # put your corpus here
ROI_DIR = os.path.join(CORPUS_DIR, "rois")   # mouth-crop cache (prep.py writes here)
RAW_DIR = os.path.join(CORPUS_DIR, "raw")    # raw video files referenced by the split files

SPM_DIR = os.path.join(BASE, "vsr", "spm")
SP_MODEL = os.path.join(SPM_DIR, "unigram5000.model")
WEIGHTS_DIR = os.path.join(BASE, "vsr", "weights")

# --- preprocessing ---
CROP_SIZE = 96
N_FRAMES = 75          # clips are padded/truncated to this many frames
SPLITS = ["train", "val", "test"]

# --- SentencePiece ---
VOCAB_SIZE = 5000      # subword vocabulary (bigger = more English coverage)

# --- Conformer model ---
ADIM = 256
AHEADS = 4
LINEAR_UNITS = 2048
ENC_LAYERS = 6
DROPOUT = 0.1

# --- training ---
BATCH_SIZE = 8
BASE_LR = 1e-4
MAX_EPOCH = 200
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
DISPLAY_ITERS = 50
