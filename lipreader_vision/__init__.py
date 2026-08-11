"""LipReader vision helpers: MediaPipe face-landmark detection + mean-face template.

The 4-keypoint landmark detector and the mean-face alignment template are vendored
from the Chaplin reference project (Imperial College London, Pingchuan Ma, Apache 2.0)
because training and live inference share this exact preprocessing pipeline.
"""

from .detector import LandmarksDetector

__all__ = ["LandmarksDetector"]
