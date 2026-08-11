"""Extract downloaded GRID archives and build train/val/test clip index files.

Usage:
    python prep.py
"""

import os
import sys
import tarfile
import zipfile

from config import ALIGN_EXTRACT_DIR, DATA_DIR, RAW_DIR, TEST_SPEAKERS, TRAIN_SPEAKERS, VIDEO_EXTRACT_DIR

VIDEO_ZIP_DIR = os.path.join(RAW_DIR, "video")
ALIGN_TAR_DIR = os.path.join(RAW_DIR, "align")


def extract_speaker(s: int) -> None:
    vid_zip = os.path.join(VIDEO_ZIP_DIR, f"s{s}.mpg_vcd.zip")
    if os.path.isfile(vid_zip):
        out_dir = os.path.join(VIDEO_EXTRACT_DIR, f"s{s}")
        if not os.path.isdir(out_dir):
            try:
                print(f"extracting video s{s} ...")
                with zipfile.ZipFile(vid_zip) as z:
                    z.extractall(out_dir)
            except zipfile.BadZipFile:
                print(f"  skipping s{s}: video zip incomplete")

    align_tar = os.path.join(ALIGN_TAR_DIR, f"s{s}.tar")
    if os.path.isfile(align_tar):
        out_dir = os.path.join(ALIGN_EXTRACT_DIR, f"s{s}")
        if not os.path.isdir(out_dir):
            try:
                print(f"extracting align s{s} ...")
                with tarfile.open(align_tar) as t:
                    t.extractall(out_dir)
            except tarfile.ReadError:
                print(f"  skipping s{s}: align tar incomplete")


def find_clip_files(s: int) -> list[tuple[str, str]]:
    """Return list of (video_path, align_path) for one speaker."""
    vroot = os.path.join(VIDEO_EXTRACT_DIR, f"s{s}")
    aroot = os.path.join(ALIGN_EXTRACT_DIR, f"s{s}")
    clips = []
    if not os.path.isdir(vroot):
        return clips
    for vdir, _, files in os.walk(vroot):
        for f in files:
            if not f.endswith(".mpg"):
                continue
            clip = os.path.splitext(f)[0]
            video = os.path.join(vdir, f)
            align = None
            for adir, _, afiles in os.walk(aroot):
                if f"{clip}.align" in afiles:
                    align = os.path.join(adir, f"{clip}.align")
                    break
            clips.append((video, align, f"s{s}/{clip}"))
    return clips


def read_label(align_path: str | None) -> str:
    if not align_path or not os.path.isfile(align_path):
        return ""
    words = []
    with open(align_path, encoding="latin-1") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3 and parts[2] not in ("sil", "sp"):
                words.append(parts[2])
    return " ".join(words)


def build_index() -> None:
    all_clips = []
    for s in sorted(set(TRAIN_SPEAKERS + TEST_SPEAKERS)):
        extract_speaker(s)
        all_clips.extend(find_clip_files(s))
    print(f"total clips found: {len(all_clips)}")

    train, val, test = [], [], []
    for video, align, clip_id in sorted(all_clips):
        label = read_label(align)
        if not label:
            print(f"skipping {clip_id}: no label")
            continue
        s_num = int(clip_id.split("/")[0][1:])
        line = f"{clip_id}\t{video}\t{align}\t{label}\n"
        if s_num in TEST_SPEAKERS:
            test.append(line)
        else:
            # hold out 10% of train speakers' clips for validation
            if hash(clip_id) % 10 == 0:
                val.append(line)
            else:
                train.append(line)

    with open(os.path.join(DATA_DIR, "train.txt"), "w", encoding="utf-8") as f:
        f.writelines(train)
    with open(os.path.join(DATA_DIR, "val.txt"), "w", encoding="utf-8") as f:
        f.writelines(val)
    with open(os.path.join(DATA_DIR, "test.txt"), "w", encoding="utf-8") as f:
        f.writelines(test)
    print(f"train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    build_index()
