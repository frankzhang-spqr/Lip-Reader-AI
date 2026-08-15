# Datasets for the open-vocabulary Conformer trainer

This is the "train your own big model" path. **It is only worth doing if you
want to reproduce research.** The pretrained AutoAVSR English reader
(`--model english`) already does better than anything you can train on a single
RTX 4060.

## What the real models are trained on

| Dataset | Hours | Licence | Access |
|---|---|---|---|
| LRS2 (BBC) | 225 | BBC Research licence | https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html (fill the form) |
| LRS3 (TED) | 439 | Research licence | https://mmai.io/datasets/lip_reading/ (fill the form) |
| VoxCeleb2 | ~2,300 | Research, YouTube clips | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html |
| AVSpeech | ~4,700 | Research, YouTube clips | https://looking-to-listen.github.io/avspeech/download.html |

## Recommendation

Start with **LRS3** (best quality/size ratio) plus **VoxCeleb2** if you want
more data. Download the raw videos, then point this pipeline at them.

## After you download

1. Put videos somewhere, e.g. `corpus/raw/`.

2. Create the split files. Format is one clip per line, TAB-separated:
   ```
   clip_id<TAB>video_path<TAB>text
   ```
   Example `corpus/train.txt`:
   ```
   s1_001	clip0001.mp4	this is a test sentence
   s1_002	clip0002.mp4	another example sentence
   ```
   `video_path` is relative to `corpus/raw/` (or absolute). Make three files:
   `train.txt`, `val.txt`, `test.txt`.

3. Build the mouth-crop ROI cache (one time; runs FaceMesh 468-pt detection
   over every frame):
   ```powershell
   python vsr/prep.py --workers 4
   ```

4. Train (SentencePiece tokenizer is trained automatically on first run):
   ```powershell
   python vsr/train.py --epochs 200
   ```

## Honest expectations

- On a single RTX 4060 (8 GB), even the small LRS3-only Conformer will take
  weeks and land at roughly **36% WER** — the pretrained AutoAVSR reader gets
  ~20% today for free.
- Reaching SOTA (~20% WER) needs the full LRS3+VoxCeleb2+AVSpeech stack and
  many GPUs (the published models used 8x A100 for weeks).
- Best use of this trainer: **fine-tune the AutoAVSR checkpoint** on your own
  domain data (e.g. your recorded clips) rather than train from scratch.
