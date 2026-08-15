"""MediaPipe FaceMesh (468-landmark) face tracking.

Higher-fidelity than the 4-keypoint detector: returns the full 468-point mesh,
and can derive the 4 stable alignment points (right eye, left eye, nose, mouth)
used by the affine mouth-crop pipeline.

Vendored MediaPipe usage (Google, Apache 2.0).
"""

import mediapipe as mp
import numpy as np

# FaceMesh landmark indices used to derive the 4-point quad.
_RIGHT_EYE = [33, 133, 159, 145]
_LEFT_EYE = [362, 263, 386, 374]
_NOSE = [1, 4]
# outer mouth contour (MediaPipe indices), mean = mouth center
_MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0]


class FaceMeshDetector:
    """Per-frame MediaPipe FaceMesh detection returning 468 landmarks."""

    def __init__(self, refine_landmarks: bool = True, min_detection_confidence: float = 0.5):
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """Return (468, 2) pixel landmarks for one RGB frame, or None."""
        results = self._face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None
        h, w = frame_rgb.shape[:2]
        lm = results.multi_face_landmarks[0].landmark
        return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float64)

    @staticmethod
    def quad_points(landmarks: np.ndarray) -> np.ndarray:
        """Collapse a (468, 2) mesh into the 4 stable alignment points.

        Order matches the mean-face template used by the crop pipeline:
        [right eye, left eye, nose tip, mouth center].
        """
        return np.vstack(
            [
                landmarks[_RIGHT_EYE].mean(axis=0),
                landmarks[_LEFT_EYE].mean(axis=0),
                landmarks[_NOSE].mean(axis=0),
                landmarks[_MOUTH].mean(axis=0),
            ]
        )
