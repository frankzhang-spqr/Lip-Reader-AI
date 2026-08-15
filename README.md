# LipReader

Real-time visual speech recognition (lip-reading) from a webcam, running fully
locally, with **two models**:

1. **GRID LipNet** — trained from scratch by you on the GRID corpus (34 speakers,
   ~51-word vocabulary) using a PyTorch LipNet (Conv3D + BiGRU + CTC).
2. **Open-vocabulary English reader** — the pretrained **AutoAVSR** model
   (~250M params, ResNet3D + Conformer + Transformer, trained on 3,300 h of
   LRS2/LRS3/VoxCeleb2/AVSpeech). Decodes **arbitrary English sentences** via a
   5000-subword vocabulary with beam search + an English language model.

Both use a MediaPipe **FaceMesh** face tracker (468 points) with an affine
mouth-crop pipeline shared between training and live inference.

## Environment

Everything runs in the conda env **`lipreader`** (Python 3.12, CUDA 12.6):

```powershell
conda activate lipreader
```

Main packages: `torch 2.13+cu126`, `torchvision`, `mediapipe 0.10.21`,
`opencv-contrib-python 4.11`, `numpy`, `scipy`, `scikit-image`, `editdistance`,
`tqdm`, `huggingface_hub`.

## Step 0 — Open-vocabulary English reader (no training needed)

The English reader's weights (~1.2 GB) auto-download from Hugging Face on first
use, or run it manually:

```powershell
python download_english_models.py
python live_lipreader.py --model english
```

## Step 1 — Data (one time, only for training the GRID LipNet)

GRID corpus video + alignments are downloaded automatically for all 33
available speakers into `training/raw/`:

```powershell
powershell -ExecutionPolicy Bypass -File training\download_grid.ps1
```

Extract archives, build train/val/test index files:

```powershell
python training\prep.py
```

Preprocess every clip into mouth-crop sequences (cached as `.npy` files in
`training/cache/`, ~30k clips):

```powershell
python -c "import sys; sys.path.insert(0, r'training'); from dataset import build_cache; [build_cache(r'training\data\'+n+'.txt', num_workers=16) for n in ('train','val','test')]"
```

> Cache is already built for this machine, so you can skip these first steps.

## Step 2 — Train your own model

```powershell
python training\train.py --epochs 200
```

- CTC loss, AdamW, batch 16, 75-frame clips, 96x96 grayscale mouth crops.
- A tqdm progress bar per epoch shows progress, ETA, iterations/s, and live
  batch + running-average loss; checkpoints saved to
  `training/weights/lipnet_latest.pt` (best val WER → `lipnet_best.pt`).
- **Stop & resume safely:** press **Ctrl+C** anytime — progress is saved to
  `training/weights/checkpoint.pt` and the run exits cleanly. Re-run the same
  command and it auto-resumes from the best checkpoint
  (`checkpoint_best.pt`, else `checkpoint.pt`), continuing at the next epoch
  with the optimizer state intact. Use `--fresh` to start over, or
  `--resume-from PATH` to pick a specific checkpoint.
- Expect ~5-10 min per epoch on the RTX 4060. Converges noticeably after
  ~50-100 epochs (GRID WER ~10-20% after full training).
- Monitor: `tensorboard --logdir logs` (optional).

Useful flags:

```powershell
python training\train.py --epochs 5          # quick sanity run
python training\train.py --epochs 200 --lr 1e-4 --batch-size 16
```

## Step 3 — Run it live

```powershell
python live_lipreader.py                     # GRID LipNet (fast, 51-word vocab)
python live_lipreader.py --model english     # open-vocabulary English (AutoAVSR)
```

- Grid model uses `training/weights/lipnet_best.pt` (falls back to
  `lipnet_latest.pt`); the English model auto-downloads its weights.
- **SPACE** → start recording, say a sentence clearly at the camera (good
  light, straight-on face, lips visible).
- **SPACE** again → decode; the transcript appears in the window (and prints
  to the terminal). Decoding runs in a background thread — the English model
  takes a few seconds per sentence.
- **Q** → quit. The window is auto-sized to fit your laptop screen.

### Options

```powershell
python live_lipreader.py --model english --cpu   # English reader on CPU (slow)
python live_lipreader.py --demo 60               # capture 60 frames, decode, print, exit
```

## Accuracy notes (read this)

- The **GRID LipNet** reads a ~51-word vocabulary (digits, colors, letters,
  commands, prepositions). It will NOT read arbitrary English words.
- The **English reader** (AutoAVSR) decodes unrestricted English. It is the
  state of the art in visual-only speech recognition: ~19-25% word error on
  the LRS3 benchmark, i.e. ~75-80% word accuracy on clear, frontal, well-lit
  speech. Casual conversation / poor light drops this substantially.
- Honest limit: **99% word accuracy on 90% of the English dictionary does not
  exist in any lip-reading system today** — many phonemes are visually
  identical (e.g. p/b/m). The English reader + its language model is the
  closest any fully-local model gets.

## Train a bigger Conformer yourself (optional)

`vsr/` is a from-scratch open-vocabulary trainer (Conv3D ResNet + Conformer +
CTC, SentencePiece subwords) for when you have a large corpus:

```powershell
python vsr/prep.py          # build mouth-crop ROIs from raw videos (FaceMesh)
python vsr/train.py --epochs 200
```

See **`vsr/DATA.md`** for how to obtain LRS3 / LRS2 / VoxCeleb2 and honest
expectations for single-GPU training.

## Files

| File | Purpose |
|---|---|
| `live_lipreader.py` | Live webcam app (`--model grid` \| `english`) |
| `lipnet_reader.py` | Inference wrapper for the trained GRID LipNet |
| `english_reader.py` | Open-vocabulary English reader (pretrained AutoAVSR) |
| `download_english_models.py` | Downloads AutoAVSR weights (~1.2 GB, HF) |
| `autoavsr/` | Vendored AutoAVSR inference core (espnet backend, Apache-2.0) |
| `training/download_grid.ps1` | Downloads GRID corpus (all speakers) |
| `training/prep.py` | Extracts archives, builds train/val/test index |
| `training/dataset.py` | Mouth-crop cache builder + PyTorch Dataset |
| `training/model.py` | LipNet model + greedy CTC decoder |
| `training/train.py` | CTC training loop (tqdm, resume, Ctrl+C-safe) |
| `training/config.py` | Vocabulary + hyperparameters |
| `lipreader_vision/` | FaceMesh (468-pt) + 4-keypoint detection, mean-face template |
| `vsr/` | From-scratch Conformer VSR trainer + `DATA.md` dataset guide |
| `download_english_models.py` | Downloads AutoAVSR weights (~1.2 GB, HF) |
