"""Download the AutoAVSR open-vocabulary VSR model + English RNNLM from Hugging Face.

Saves to:
    models/LRS3_V_WER19.1/   (VSR model, ~950 MB, 19.1% WER on LRS3)
    models/lm_en_subword/    (English language model for beam search)

Usage:
    python download_english_models.py
"""

import os
import sys

from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")

REPOS = {
    "Amanvir/LRS3_V_WER19.1": "LRS3_V_WER19.1",
    "Amanvir/lm_en_subword": "lm_en_subword",
}

FILES = ["model.json", "model.pth"]


def ensure_downloaded(silent: bool = False) -> None:
    """Download any missing model files. Safe to call on every startup."""
    missing = []
    for repo, local_dir in REPOS.items():
        dest = os.path.join(MODELS_DIR, local_dir)
        for fname in FILES:
            out = os.path.join(dest, fname)
            if not (os.path.isfile(out) and os.path.getsize(out) > 0):
                missing.append((repo, dest, fname))
    if not missing:
        return
    if not silent:
        print(
            "English (AutoAVSR) model files missing — downloading ~1 GB from "
            "Hugging Face (one time)."
        )
    for repo, dest, fname in missing:
        os.makedirs(dest, exist_ok=True)
        print(f"  downloading {repo}/{fname} ...", flush=True)
        hf_hub_download(repo_id=repo, filename=fname, local_dir=dest)
        print(f"  {fname}: done", flush=True)


def main():
    ensure_downloaded()
    print("\nAll model files ready.")


if __name__ == "__main__":
    sys.exit(main())
