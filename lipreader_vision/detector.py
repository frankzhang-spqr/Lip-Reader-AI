"""MediaPipe-based facial landmarks detector for lip-reading.

Vendored from the Chaplin reference project (Pingchuan Ma, Imperial College London,
Apache 2.0). Detects 4 key facial points (right eye, left eye, nose tip, mouth
center) used for affine face alignment in both training and live inference.
"""

import mediapipe as mp

import numpy as np


class LandmarksDetector:
    """MediaPipe-based facial landmarks detector.

    Uses MediaPipe face detection to extract 4 key facial landmarks from video
    frames. Falls back between short-range and full-range detection models for
    robustness.
    """

    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.short_range_detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5, model_selection=0
        )
        self.full_range_detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5, model_selection=1
        )

    def detect(self, video_frames, detector):
        """Run face detection on video frames using the given detector.

        Args:
            video_frames: NumPy array of video frames (T, H, W, C).
            detector: MediaPipe FaceDetection instance to use.

        Returns:
            List of (4, 2) landmark arrays per frame, or None for frames
            without a detected face.
        """
        landmarks = []
        for frame in video_frames:
            results = detector.process(frame)
            if not results.detections:
                landmarks.append(None)
                continue
            face_points = []
            for idx, detected_faces in enumerate(results.detections):
                max_id, max_size = 0, 0
                bboxC = detected_faces.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )
                bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                if bbox_size > max_size:
                    max_id, max_size = idx, bbox_size
                lmx = [
                    [
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(0).value
                            ].x
                            * iw
                        ),
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(0).value
                            ].y
                            * ih
                        ),
                    ],
                    [
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(1).value
                            ].x
                            * iw
                        ),
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(1).value
                            ].y
                            * ih
                        ),
                    ],
                    [
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(2).value
                            ].x
                            * iw
                        ),
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(2).value
                            ].y
                            * ih
                        ),
                    ],
                    [
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(3).value
                            ].x
                            * iw
                        ),
                        int(
                            detected_faces.location_data.relative_keypoints[
                                self.mp_face_detection.FaceKeyPoint(3).value
                            ].y
                            * ih
                        ),
                    ],
                ]
                face_points.append(lmx)
            landmarks.append(np.array(face_points[max_id]))
        return landmarks
