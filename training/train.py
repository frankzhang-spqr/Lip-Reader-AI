"""Train LipNet on GRID with CTC loss.

Usage:
    python train.py --epochs 5            # quick validation run
    python train.py --epochs 300 --build-cache   # full run (builds cache first)

Resume / interrupt:
    Ctrl+C saves progress to weights/checkpoint.pt and exits cleanly.
    Re-running the same command auto-resumes from the best checkpoint
    (weights/checkpoint_best.pt, else weights/checkpoint.pt).
    Use --fresh to start over, or --resume-from PATH to pick a file.
"""

import argparse
import os
import random

import editdistance
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from config import BLANK_IDX, IDX2CHAR, VOCAB_SIZE
from dataset import GRIDDataset, build_cache
from model import LipNet, greedy_decode


def collate(batch):
    videos, labels, lens, texts = zip(*batch)
    videos = torch.stack(videos)  # (B, T, 1, 96, 96)
    videos = videos.permute(0, 2, 1, 3, 4).contiguous()  # (B, 1, T, 96, 96)
    max_len = max(l.size(0) for l in labels)
    padded = torch.zeros(len(labels), max_len, dtype=torch.long)
    tgt_len = torch.zeros(len(labels), dtype=torch.long)
    for i, l in enumerate(labels):
        padded[i, : l.size(0)] = l
        tgt_len[i] = l.size(0)
    inp_len = torch.full((len(labels),), videos.size(2), dtype=torch.long)
    return videos, padded, inp_len, tgt_len, texts


def evaluate(model, loader, device):
    model.eval()
    cer, wer, n = 0.0, 0.0, 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="  val", unit="it", ncols=110, dynamic_ncols=False, leave=False)
        for videos, _, _, _, texts in pbar:
            videos = videos.to(device)
            logits = model(videos)
            preds = greedy_decode(logits, IDX2CHAR)
            for pred, ref in zip(preds, texts):
                ref = ref.lower().replace(" ", "")
                pred_flat = pred.replace(" ", "")
                cer += editdistance.eval(pred_flat, ref)
                ref_words = ref.replace(" ", " ").split()
                pred_words = pred.split()
                wer += editdistance.eval(pred_words, ref_words)
                n += 1
        pbar.close()
    model.train()
    if n == 0:
        return float("nan"), float("nan")
    return cer / n, wer / n


def save_checkpoint(path, model, optimizer, epoch, global_step, best_wer):
    """Save a full training state (model + optimizer + progress) for resuming."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_wer": best_wer,
        },
        path,
    )


def find_resume_checkpoint(weights_dir):
    """Pick the checkpoint to resume from: best-epoch state first, else latest."""
    best = os.path.join(weights_dir, "checkpoint_best.pt")
    latest = os.path.join(weights_dir, "checkpoint.pt")
    if os.path.isfile(best):
        return best
    if os.path.isfile(latest):
        return latest
    return None


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.build_cache:
        print("building cache...")
        build_cache(os.path.join(config.DATA_DIR, "train.txt"))
        build_cache(os.path.join(config.DATA_DIR, "val.txt"))
        build_cache(os.path.join(config.DATA_DIR, "test.txt"))

    device = "cpu"
    if torch.cuda.is_available() and not args.cpu:
        device = f"cuda:{args.gpu}"
    print(f"device: {device}")

    train_ds = GRIDDataset(os.path.join(config.DATA_DIR, "train.txt"), vid_padding=config.VID_PADDING)
    val_ds = GRIDDataset(os.path.join(config.DATA_DIR, "val.txt"), vid_padding=config.VID_PADDING)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=config.NUM_WORKERS,
        collate_fn=collate, persistent_workers=config.NUM_WORKERS > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=config.NUM_WORKERS,
        collate_fn=collate, persistent_workers=config.NUM_WORKERS > 0,
    )
    print(f"train clips: {len(train_ds)}, val clips: {len(val_ds)}")

    weights_dir = config.WEIGHTS_DIR
    model = LipNet(in_channels=1, vocab_size=VOCAB_SIZE, dropout_p=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY
    )
    steps_per_epoch = len(train_loader)
    start_epoch, global_step, best_wer = 1, 0, 999.0

    resume_path = args.resume_from
    if resume_path is None and not args.fresh:
        resume_path = find_resume_checkpoint(weights_dir)
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt["global_step"])
        best_wer = float(ckpt["best_wer"])
        print(
            f"resumed from {resume_path}: continuing at epoch {start_epoch}, "
            f"best_wer {best_wer:.4f} (use --fresh to start over)"
        )
    elif os.path.isfile(os.path.join(weights_dir, "checkpoint.pt")):
        print("--fresh: ignoring previous checkpoint, training from scratch")

    if start_epoch > args.epochs:
        print(f"already trained through epoch {args.epochs}; nothing to do (raise --epochs)")
        return

    epoch = start_epoch
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            running_loss = 0.0
            pbar = tqdm(
                train_loader,
                desc=f"epoch {epoch}/{args.epochs}",
                unit="it",
                ncols=110,
                dynamic_ncols=False,
                leave=True,
            )
            for i, (videos, padded, inp_len, tgt_len, _) in enumerate(pbar):
                videos = videos.to(device)
                padded = padded.to(device)
                inp_len = inp_len.to(device)
                tgt_len = tgt_len.to(device)

                logits = model(videos)  # (B, T, vocab)
                log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, vocab)
                loss = F.ctc_loss(
                    log_probs, padded, inp_len, tgt_len, blank=BLANK_IDX, reduction="mean"
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                running_loss += loss.item()
                global_step += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running_loss / (i + 1):.4f}")
            pbar.close()

            torch.save(model.state_dict(), os.path.join(weights_dir, "lipnet_latest.pt"))
            val_cer, val_wer = evaluate(model, val_loader, device)
            print(
                f"\n== epoch {epoch} done, train_loss {running_loss/len(train_loader):.4f} "
                f"val_cer {val_cer:.4f} val_wer {val_wer:.4f}\n"
            )
            if val_wer < best_wer:
                best_wer = val_wer
                torch.save(model.state_dict(), os.path.join(weights_dir, "lipnet_best.pt"))
                save_checkpoint(
                    os.path.join(weights_dir, "checkpoint_best.pt"),
                    model, optimizer, epoch, global_step, best_wer,
                )
            save_checkpoint(
                os.path.join(weights_dir, "checkpoint.pt"),
                model, optimizer, epoch, global_step, best_wer,
            )
    except KeyboardInterrupt:
        try:
            pbar.close()
        except Exception:  # noqa: BLE001
            pass
        save_checkpoint(
            os.path.join(weights_dir, "checkpoint.pt"),
            model, optimizer, epoch, global_step, best_wer,
        )
        print(
            "\nCtrl+C received. Progress saved to checkpoint.pt "
            f"(epoch {epoch}, best_wer {best_wer:.4f}). "
            "Re-run the same command to resume training."
        )
        return

    print("training finished")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCH)
    parser.add_argument("--lr", type=float, default=config.BASE_LR)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--display", type=int, default=config.DISPLAY_ITERS)
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=config.SAVE_EVERY_ITERS)
    parser.add_argument("--max-checkpoint-epochs", type=int, default=0,
                        help="if 0, checkpoint only at epoch end (default)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="resume from this checkpoint file (default: auto-resume from checkpoint_best.pt / checkpoint.pt)")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore saved checkpoints and start training from scratch")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
