"""Live lip-reading from webcam using the AutoAVSR visual speech model.

Usage:
    python live_lipreader.py                # webcam, CUDA if available
    python live_lipreader.py --cpu           # force CPU
    python live_lipreader.py --beam-size 20  # faster decode, slightly less accurate
    python live_lipreader.py --demo 30       # capture 30 frames, decode, print, exit (no GUI)

Controls (in the camera window):
    SPACE  toggle recording  (hold to record a sentence, release to decode)
    Q      quit
"""

import argparse
import os
import threading
import time

import cv2
import numpy as np
import torch

from lipnet_reader import LipNetReader

TRAIN_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training", "weights")


def load_reader(device: str) -> LipNetReader:
    """Load our trained LipNet if present, otherwise error with instructions."""
    for name in ("lipnet_best.pt", "lipnet_latest.pt"):
        wpath = os.path.join(TRAIN_WEIGHTS_DIR, name)
        if os.path.isfile(wpath):
            print(f"loading locally trained LipNet: {name}")
            return LipNetReader(wpath, device=device)
    raise FileNotFoundError(
        "No trained model found. Train one first:\n"
        "    conda activate lipreader\n"
        "    python training/train.py --build-cache --epochs 200"
    )


def draw_keypoints(frame_bgr: np.ndarray, lm: np.ndarray | None, color: tuple) -> None:
    if lm is not None:
        for (x, y) in lm.astype(int):
            cv2.circle(frame_bgr, (int(x), int(y)), 4, color, -1)


def infer_async(reader: LipNetReader, frames: list, landmarks: list, result: list) -> None:
    try:
        text = reader.process_sequence(frames, landmarks)
        result[0] = text or "(no face detected)"
    except Exception as exc:  # noqa: BLE001 - surface any pipeline error to the user
        result[0] = f"error: {exc}"
    finally:
        result[1] = False


def run_live(reader: LipReader, camera_index: int, max_frames: int | None = None) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}. Is a webcam connected?")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        import ctypes

        user32 = ctypes.windll.user32
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
    except Exception:  # noqa: BLE001 - fallback if screen size is unavailable
        screen_w, screen_h = 1366, 768
    win_w = min(800, screen_w - 60)
    win_h = min(600, screen_h - 80)
    cv2.namedWindow("LipReader", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("LipReader", win_w, win_h)

    recording = False
    frames: list = []
    landmarks: list = []
    result = ["", False]  # [text, busy]
    thread = None
    n = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_bgr = cv2.resize(frame_bgr, (640, 480))
            frame_bgr = cv2.flip(frame_bgr, 1)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            lm = reader.detect_frame(rgb)
            n += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and not result[1]:
                recording = not recording
                if recording:
                    frames, landmarks = [], []
                elif frames:
                    result[0] = "decoding..."
                    result[1] = True
                    thread = threading.Thread(
                        target=infer_async,
                        args=(reader, list(frames), list(landmarks), result),
                        daemon=True,
                    )
                    thread.start()

            if recording:
                frames.append(rgb)
                landmarks.append(lm)
                color = (0, 200, 0)
                status = f"RECORDING {len(frames)} frames (press SPACE to decode)"
            elif result[1]:
                color = (0, 165, 255)
                status = "decoding..."
            else:
                color = (0, 0, 200)
                status = "Press SPACE to record a sentence"

            draw_keypoints(frame_bgr, lm, color)
            cv2.putText(
                frame_bgr, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
            transcript = result[0]
            if transcript:
                cv2.putText(
                    frame_bgr,
                    transcript,
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
            cv2.imshow("LipReader", frame_bgr)

            if max_frames and n >= max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_demo(reader: LipReader, camera_index: int, num_frames: int) -> None:
    """Capture num_frames from the camera, decode, print the result, exit."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}. Is a webcam connected?")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frames, landmarks = [], []
    print(f"capturing {num_frames} frames from camera {camera_index}...")
    for _ in range(num_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        landmarks.append(reader.detect_frame(rgb))
    cap.release()
    if not frames:
        print("no frames captured")
        return
    print(f"decoding {len(frames)} frames...")
    t0 = time.time()
    text = reader.process_sequence(frames, landmarks)
    print(f"decoded in {time.time() - t0:.1f}s -> {text!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live lip-reading from webcam")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    parser.add_argument("--demo", type=int, metavar="N", help="capture N frames, decode, exit")
    parser.add_argument("--live-frames", type=int, metavar="N", help="run GUI for N frames then exit")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0"
    print(f"loading model on {device}...")
    reader = load_reader(device)
    print("model ready")

    if args.demo:
        run_demo(reader, args.camera, args.demo)
    else:
        run_live(reader, args.camera, max_frames=args.live_frames)


if __name__ == "__main__":
    main()
