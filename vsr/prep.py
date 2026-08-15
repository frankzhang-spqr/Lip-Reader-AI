"""Build mouth-crop ROI cache for the Conformer trainer.

Preprocesses raw video clips (FaceMesh 468-pt tracking -> 4-point affine
alignment -> 96x96 grayscale crops) so training does not re-run detection.

Usage:
    python vsr/prep.py                 # build ROIs for all splits
    python vsr/prep.py --split train   # build one split only
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "autoavsr"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config  # noqa: E402
from dataset import parse_index  # noqa: E402
from lipreader_vision.facemesh import FaceMeshDetector  # noqa: E402
from autoavsr.video_process import VideoProcess  # noqa: E402


def build_clip(video_path: str, detector: FaceMeshDetector, video_process: VideoProcess) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    frames, quads = [], []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        lm = detector.detect(rgb)
        quad = detector.quad_points(lm) if lm is not None else None
        frames.append(rgb)
        quads.append(quad)
    cap.release()
    if len(frames) < 25:
        return None
    video = np.stack(frames, axis=0)
    patches = video_process(video, quads)
    if patches is None:
        return None
    return patches.astype(np.float32) / 255.0


def build_split(split: str, num_workers: int = 4) -> None:
    split_path = os.path.join(config.CORPUS_DIR, split + ".txt")
    if not os.path.isfile(split_path):
        print(f"skip {split}: no {split_path}")
        return
    os.makedirs(config.ROI_DIR, exist_ok=True)
    rows = parse_index(split_path)
    print(f"preparing {split}: {len(rows)} clips (workers={num_workers})")

    detector = FaceMeshDetector()
    video_process = VideoProcess(convert_gray=True, mean_face_path="mean_face.npy")

    def work(row):
        clip_id, video_path, _ = row
        out = os.path.join(config.ROI_DIR, clip_id + ".npy")
        if os.path.isfile(out):
            return "cached"
        full = video_path if os.path.isabs(video_path) else os.path.join(config.RAW_DIR, video_path)
        patches = build_clip(full, detector, video_process)
        if patches is None:
            return "failed"
        np.save(out, patches)
        return "ok"

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        for fut in as_completed([ex.submit(work, r) for r in rows]):
            res = fut.result()
            if res == "ok":
                ok += 1
            elif res == "failed":
                fail += 1
    print(f"{split}: ok={ok} fail={fail} cached=rest")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=config.SPLITS, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    splits = [args.split] if args.split else config.SPLITS
    for s in splits:
        build_split(s, args.workers)


if __name__ == "__main__":
    main()
