"""
Unified Sign Language Vision & Inference Engine.
Connects MediaPipe Hand Landmarking with Deep Learning (ResNet, LSTM)
and Classical Machine Learning (Random Forest) models.

Author: CSAI3027P Student Suite
Purpose: Real-time hand landmark extraction, skeleton drawing, and multi-model inference.
"""

import os
from collections import Counter, deque
from typing import Dict, List, Optional, Tuple
import cv2
import joblib
import numpy as np
import torch

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

try:
    import keras
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False

from src.models.ml_classifier import SignMLClassifier
from src.vision.resnet_model import SignResNet

# Hand skeleton connections for landmark visualization
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palm base
    (0, 17),
]


class SignInferenceEngine:
    """
    Unified Inference Engine coordinating MediaPipe Landmark Detection,
    Deep Learning (ResNet-1D & Keras LSTM), and Classical Machine Learning (Random Forest).
    """

    def __init__(
        self,
        base_dir: str = "/Users/gauravgoyal/Desktop/Sign_Language",
        sequence_length: int = 25,
        min_confidence: float = 0.35,
    ):
        self.base_dir = base_dir
        self.seq_len = sequence_length
        self.min_confidence = min_confidence

        # Load class labels
        classes_path = os.path.join(base_dir, "classes.npy")
        if os.path.exists(classes_path):
            self.classes = list(np.load(classes_path, allow_pickle=True))
        else:
            self.classes = [f"sign_{i}" for i in range(98)]
        self.n_classes = len(self.classes)

        # Lazy-loaded model instances
        self._hand_detector = None
        self._resnet_model = None
        self._lstm_model = None
        self._ml_model = None

    def get_hand_detector(self):
        """Initializes and caches the MediaPipe HandLandmarker."""
        if self._hand_detector is None and HAS_MEDIAPIPE:
            model_task = os.path.join(self.base_dir, "hand_landmarker.task")
            if os.path.exists(model_task):
                options = vision.HandLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=model_task),
                    num_hands=2,
                )
                self._hand_detector = vision.HandLandmarker.create_from_options(options)
        return self._hand_detector

    def get_resnet_model(self) -> SignResNet:
        """Loads and caches the PyTorch SignResNet Deep Learning model."""
        if self._resnet_model is None:
            model = SignResNet(in_features=126, num_classes=self.n_classes)
            weights_path = os.path.join(self.base_dir, "sign_resnet_weights.pt")
            if os.path.exists(weights_path):
                state = torch.load(weights_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
            model.eval()
            self._resnet_model = model
        return self._resnet_model

    def get_lstm_model(self):
        """Loads and caches the Keras LSTM Deep Learning model."""
        if self._lstm_model is None and HAS_KERAS:
            lstm_path = os.path.join(self.base_dir, "sign_language_model_100.keras")
            if os.path.exists(lstm_path):
                self._lstm_model = keras.models.load_model(lstm_path)
        return self._lstm_model

    def get_ml_model(self) -> SignMLClassifier:
        """Loads and caches the Random Forest Classical Machine Learning model."""
        if self._ml_model is None:
            ml_path = os.path.join(self.base_dir, "sign_ml_model.joblib")
            self._ml_model = SignMLClassifier(ml_path)
        return self._ml_model

    def extract_landmarks(
        self, frame: np.ndarray, draw_overlay: bool = True
    ) -> Tuple[List[float], np.ndarray, int]:
        """
        Extracts 126 wrist-relative coordinates from a video frame
        and optionally draws the glowing skeletal overlay.
        
        Returns:
            (landmarks_126, annotated_frame, detected_hands_count)
        """
        detector = self.get_hand_detector()
        left_hand = [0.0] * 63
        right_hand = [0.0] * 63
        hands_detected = 0

        annotated_frame = frame.copy()
        if detector is None:
            return left_hand + right_hand, annotated_frame, 0

        # Convert to MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        h, w, _ = frame.shape

        if result and result.hand_landmarks:
            hands_detected = len(result.hand_landmarks)
            for i, hand in enumerate(result.hand_landmarks):
                wrist = hand[0]
                values = []
                coords_2d = []

                for pt in hand:
                    values.extend([pt.x - wrist.x, pt.y - wrist.y, pt.z - wrist.z])
                    coords_2d.append((int(pt.x * w), int(pt.y * h)))

                # Scale normalization
                biggest = max([abs(v) for v in values], default=0.0)
                if biggest > 0.0:
                    values = [v / biggest for v in values]

                side = result.handedness[i][0].category_name
                if side == "Left":
                    left_hand = values
                    bone_color = (255, 180, 79)   # Cyan / Blue
                    joint_color = (255, 255, 255) # White
                else:
                    right_hand = values
                    bone_color = (255, 123, 155)  # Violet / Magenta
                    joint_color = (255, 255, 255) # White

                # Draw skeleton lines and joints if requested
                if draw_overlay:
                    for start_idx, end_idx in HAND_CONNECTIONS:
                        if start_idx < len(coords_2d) and end_idx < len(coords_2d):
                            pt1 = coords_2d[start_idx]
                            pt2 = coords_2d[end_idx]
                            cv2.line(annotated_frame, pt1, pt2, bone_color, 2, cv2.LINE_AA)

                    for pt in coords_2d:
                        cv2.circle(annotated_frame, pt, 4, joint_color, -1, cv2.LINE_AA)
                        cv2.circle(annotated_frame, pt, 5, bone_color, 1, cv2.LINE_AA)

        return left_hand + right_hand, annotated_frame, hands_detected

    def predict_sequence(
        self, sequence: np.ndarray, model_type: str = "ResNet (Deep Learning)"
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predict one 25-frame landmark sequence.

        Important: use the same canonical camera orientation used by landmark
        extraction. Do not select the most confident result from artificial
        flips/swaps, because that can turn an incorrect class into a confident
        prediction.
        """
        seq = np.asarray(sequence, dtype=np.float32)

        if seq.ndim != 2 or seq.shape[1] != 126:
            raise ValueError("Expected landmark sequence with shape (frames, 126).")

        if not np.isfinite(seq).all():
            raise ValueError("Landmark sequence contains NaN or infinite values.")

        if seq.shape[0] > self.seq_len:
            indices = np.linspace(0, seq.shape[0] - 1, self.seq_len).round().astype(int)
            seq = seq[indices]
        elif seq.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - seq.shape[0], 126), dtype=np.float32)
            seq = np.vstack([seq, pad])

        cand_batch = np.expand_dims(seq, axis=0)

        if "ResNet" in model_type:
            model = self.get_resnet_model()
            with torch.no_grad():
                tensor_in = torch.from_numpy(cand_batch).float()
                logits = model(tensor_in)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        elif "LSTM" in model_type:
            model = self.get_lstm_model()
            if model is None:
                return self.predict_sequence(seq, "ResNet (Deep Learning)")
            probs = np.asarray(model.predict(cand_batch, verbose=0)[0], dtype=np.float32)

        elif "Random Forest" in model_type:
            ml_clf = self.get_ml_model()
            features = ml_clf.extract_features(seq).reshape(1, -1)
            raw_probs = ml_clf.model.predict_proba(features)[0]
            model_classes = list(ml_clf.classes)
            probs = np.zeros(len(self.classes), dtype=np.float32)
            class_to_index = {str(name): i for i, name in enumerate(self.classes)}
            for i, name in enumerate(model_classes):
                if str(name) in class_to_index:
                    probs[class_to_index[str(name)]] = float(raw_probs[i])

        else:
            resnet = self.get_resnet_model()
            with torch.no_grad():
                tensor_in = torch.from_numpy(cand_batch).float()
                logits = resnet(tensor_in)
                p_dl = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            ml_clf = self.get_ml_model()
            features = ml_clf.extract_features(seq).reshape(1, -1)
            raw_ml = ml_clf.model.predict_proba(features)[0]
            model_classes = list(ml_clf.classes)
            p_ml = np.zeros(len(self.classes), dtype=np.float32)
            class_to_index = {str(name): i for i, name in enumerate(self.classes)}
            for i, name in enumerate(model_classes):
                if str(name) in class_to_index:
                    p_ml[class_to_index[str(name)]] = float(raw_ml[i])
            probs = 0.55 * p_dl + 0.45 * p_ml

        probs = np.asarray(probs, dtype=np.float32).reshape(-1)
        if len(probs) != len(self.classes):
            raise ValueError(
                "Model output count does not match classes.npy. Check model and class-label files."
            )

        if not np.isfinite(probs).all() or float(np.sum(probs)) <= 0:
            raise ValueError("Model returned invalid prediction probabilities.")

        probs = probs / np.sum(probs)
        top_indices = np.argsort(probs)[-5:][::-1]
        best_idx = int(top_indices[0])
        top_class = str(self.classes[best_idx])
        top_conf = float(probs[best_idx])
        top5 = [(str(self.classes[i]), float(probs[i])) for i in top_indices]

        return top_class, top_conf, top5
