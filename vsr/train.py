"""Train the open-vocabulary Conformer VSR model (CTC) on your own corpus.

Usage:
    python vsr/train.py --epochs 200                       # full run
    python vsr/train.py --epochs 5                         # sanity check
    python vsr/train.py --train-tokenizer                  # (re)build SentencePiece from corpus texts first

Resume / interrupt:
    Ctrl+C saves progress to vsr/weights/checkpoint.pt and exits cleanly.
    Re-running auto-resumes from checkpoint_best.pt (else checkpoint.pt).
    Use --fresh to start over.

Note: for real open-vocabulary results you need a large dataset (LRS3,
VoxCeleb2...) and ideally fine-tune from the pretrained AutoAVSR checkpoint.
See vsr/DATA.md.
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
from dataset import CorpusDataset, collate, parse_index
from model import VideoConformer, greedy_decode
from tokenizer import Tokenizer, train_tokenizer


def evaluate(model, loader, device, tokenizer):
    model.eval()
    cer, wer, n = 0.0, 0.0, 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="  val", unit="it", ncols=110, leave=False)
        for videos, inp_len, _, _, texts in pbar:
            videos = videos.to(device)
            logits = model(videos)
            preds = greedy_decode(logits, tokenizer.token_list)
            for pred, ref in zip(preds, texts):
                pred_words = pred.replace("▁", " ").split()
                ref_words = ref.lower().split()
                cer += editdistance.eval(list(pred.replace(" ", "")), list("".join(ref_words)))
                wer += editdistance.eval(pred_words, ref_words)
                n += 1
        pbar.close()
    model.train()
    if n == 0:
        return float("nan"), float("nan")
    return cer / n, wer / n


def save_checkpoint(path, model, optimizer, epoch, global_step, best_wer):
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


def find_resume(weights_dir):
    for name in ("checkpoint_best.pt", "checkpoint.pt"):
        p = os.path.join(weights_dir, name)
        if os.path.isfile(p):
            return p
    return None


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(config.WEIGHTS_DIR, exist_ok=True)
    os.makedirs(config.SPM_DIR, exist_ok=True)

    if args.train_tokenizer or not os.path.isfile(config.SP_MODEL):
        text_files = []
        for split in config.SPLITS:
            p = os.path.join(config.CORPUS_DIR, split + ".txt")
            if os.path.isfile(p):
                text_files.append(p)
        assert text_files, "no corpus/*.txt split files found — see vsr/DATA.md"
        print("training SentencePiece tokenizer...")
        train_tokenizer(text_files, config.SP_MODEL, config.VOCAB_SIZE)
    tokenizer = Tokenizer(config.SP_MODEL, config.VOCAB_SIZE)
    print(f"tokenizer vocab: {len(tokenizer.token_list)}")

    device = "cpu"
    if torch.cuda.is_available() and not args.cpu:
        device = f"cuda:{args.gpu}"
    print(f"device: {device}")

    train_ds = CorpusDataset(os.path.join(config.CORPUS_DIR, "train.txt"), tokenizer)
    val_ds = CorpusDataset(os.path.join(config.CORPUS_DIR, "val.txt"), tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=config.NUM_WORKERS,
        collate_fn=collate, persistent_workers=config.NUM_WORKERS > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=config.NUM_WORKERS,
        collate_fn=collate, persistent_workers=config.NUM_WORKERS > 0,
    )
    print(f"train clips: {len(train_ds)}, val clips: {len(val_ds)}")

    model = VideoConformer(
        vocab_size=len(tokenizer.token_list),
        blank=tokenizer.blank,
        adim=config.ADIM,
        aheads=config.AHEADS,
        linear_units=config.LINEAR_UNITS,
        enc_layers=config.ENC_LAYERS,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)

    start_epoch, global_step, best_wer = 1, 0, 999.0
    resume_path = args.resume_from
    if resume_path is None and not args.fresh:
        resume_path = find_resume(config.WEIGHTS_DIR)
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt["global_step"])
        best_wer = float(ckpt["best_wer"])
        print(f"resumed from {resume_path}: continuing at epoch {start_epoch}, best_wer {best_wer:.4f}")

    if start_epoch > args.epochs:
        print(f"already trained through epoch {args.epochs}; nothing to do (raise --epochs)")
        return

    epoch = start_epoch
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", unit="it", ncols=110, leave=True)
            for i, (videos, inp_len, padded, tgt_len, _) in enumerate(pbar):
                videos, inp_len = videos.to(device), inp_len.to(device)
                padded, tgt_len = padded.to(device), tgt_len.to(device)
                logits = model(videos)  # (B, T, vocab)
                log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, vocab)
                loss = F.ctc_loss(log_probs, padded, inp_len, tgt_len, blank=tokenizer.blank, reduction="mean")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                running_loss += loss.item()
                global_step += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running_loss / (i + 1):.4f}")
            pbar.close()

            torch.save(model.state_dict(), os.path.join(config.WEIGHTS_DIR, "latest.pt"))
            val_cer, val_wer = evaluate(model, val_loader, device, tokenizer)
            print(f"\n== epoch {epoch} done, train_loss {running_loss / len(train_loader):.4f} "
                  f"val_cer {val_cer:.4f} val_wer {val_wer:.4f}\n")
            if val_wer < best_wer:
                best_wer = val_wer
                torch.save(model.state_dict(), os.path.join(config.WEIGHTS_DIR, "best.pt"))
                save_checkpoint(os.path.join(config.WEIGHTS_DIR, "checkpoint_best.pt"),
                                model, optimizer, epoch, global_step, best_wer)
            save_checkpoint(os.path.join(config.WEIGHTS_DIR, "checkpoint.pt"),
                            model, optimizer, epoch, global_step, best_wer)
    except KeyboardInterrupt:
        try:
            pbar.close()
        except Exception:  # noqa: BLE001
            pass
        save_checkpoint(os.path.join(config.WEIGHTS_DIR, "checkpoint.pt"),
                        model, optimizer, epoch, global_step, best_wer)
        print(f"\nCtrl+C received. Progress saved (epoch {epoch}, best_wer {best_wer:.4f}). "
              "Re-run the same command to resume.")
        return

    print("training finished")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCH)
    parser.add_argument("--lr", type=float, default=config.BASE_LR)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--dropout", type=float, default=config.DROPOUT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-tokenizer", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
