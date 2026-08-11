# LipReader

Real-time visual speech recognition (lip-reading) from a webcam, running fully
locally. The model is **trained from scratch by you** on the **GRID corpus**
(34 speakers, sentence-level, ~51-word vocabulary) using a PyTorch **LipNet**
(Conv3D + BiGRU + CTC). The mouth crops use the same MediaPipe 4-keypoint
affine pipeline in training and in the live app.

## Environment

Everything runs in the conda env **`lipreader`** (Python 3.12, CUDA 12.6):

```powershell
conda activate lipreader
```

Main packages: `torch 2.13+cu126`, `torchvision`, `mediapipe 0.10.21`,
`opencv-contrib-python 4.11`, `numpy`, `scipy`, `editdistance`, `tqdm`.

## Step 1 — Data (one time)

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
python live_lipreader.py
```

- Uses `training/weights/lipnet_best.pt` (falls back to `lipnet_latest.pt`).
- **SPACE** → start recording, say a sentence clearly at the camera (good
  light, straight-on face, lips visible).
- **SPACE** again → decode; the transcript appears in the window (and prints
  to the terminal).
- **Q** → quit. The window is auto-sized to fit your laptop screen.

### Options

```powershell
python live_lipreader.py --cpu            # force CPU
python live_lipreader.py --demo 60        # capture 60 frames, decode, print, exit
```

## Accuracy notes (read this)

- This model reads a **~51-word GRID vocabulary** (digits, colors, letters,
  commands, prepositions). It will NOT read arbitrary English words.
- No locally-trained model on a laptop reaches 90-95% of the dictionary; that
  needs datasets like LRS2/LRS3 (large, restricted) and many GPU-hours.
- Expected quality after a full GRID run: good on clear, frontal, well-lit
  GRID-style sentences; errors increase on unseen speakers and poor conditions.
- Training on the full 33-speaker split (~26k train clips) gives the best
  results. Fewer epochs/speakers → worse WER.

## Files

| File | Purpose |
|---|---|
| `live_lipreader.py` | Live webcam app (recording + decoding + overlay) |
| `lipnet_reader.py` | Inference wrapper for the trained LipNet |
| `training/download_grid.ps1` | Downloads GRID corpus (all speakers) |
| `training/prep.py` | Extracts archives, builds train/val/test index |
| `training/dataset.py` | Mouth-crop cache builder + PyTorch Dataset |
| `training/model.py` | LipNet model + greedy CTC decoder |
| `training/train.py` | CTC training loop with CER/WER eval |
| `training/config.py` | Vocabulary + hyperparameters |
| `lipreader_vision/` | MediaPipe face-landmark detection + mean-face template |
