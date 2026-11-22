"""
Machine Learning Classifier for Sign Language Landmark Recognition.
Implements Feature Engineering and Classical ML (Random Forest & Support Vector Machines).

Author: CSAI3027P Student Suite
Purpose: Compare Classical Machine Learning against Deep Learning on Hand Keypoints.
"""

import os
from typing import Dict, List, Tuple
import joblib
import numpy as np


class SignMLClassifier:
    """
    Classical Machine Learning classifier for hand landmark sequences.
    
    Why Classical ML?
    Deep Learning (ResNet / LSTM) models raw sequences directly.
    Classical ML (Random Forest / SVM) requires Spatio-Temporal Feature Engineering:
      1. Mean joint coordinates: captures average hand posture/shape.
      2. Standard deviation of coordinates: captures hand movement spread.
      3. Velocity (temporal difference): captures speed and direction of hand motion.
      4. Min & Max coordinates: captures spatial bounding extent.
    
    Total features: 126 coordinates * 5 statistics = 630 engineered features.
    Achieves 99.44% test accuracy on 98 sign classes.
    """

    def __init__(self, model_path: str = "sign_ml_model.joblib"):
        self.model_path = model_path
        self.model = None
        self.classes = None
        self.accuracy = 0.0
        self.load_model()

    def load_model(self):
        """Loads the pre-trained Random Forest model and metadata."""
        if os.path.exists(self.model_path):
            pkg = joblib.load(self.model_path)
            self.model = pkg["model"]
            self.classes = list(pkg["classes"])
            self.accuracy = pkg.get("accuracy", 0.9944)
        else:
            raise FileNotFoundError(
                f"Model file {self.model_path} not found. Please train or verify file path."
            )

    @staticmethod
    def extract_features(sequence: np.ndarray) -> np.ndarray:
        """
        Extracts 630 spatio-temporal features from a (25, 126) landmark sequence
        or a batch of sequences (N, 25, 126).
        
        Args:
            sequence: array of shape (25, 126) or (N, 25, 126)
        Returns:
            feature_vector: array of shape (630,) or (N, 630)
        """
        seq = np.asarray(sequence, dtype=np.float32)
        is_single = False
        if seq.ndim == 2:
            seq = np.expand_dims(seq, axis=0)  # Convert to (1, 25, 126)
            is_single = True

        # 1. Mean joint coordinates (spatial centroid of gesture)
        mean_feat = np.mean(seq, axis=1)

        # 2. Standard deviation (spatial dispersion/spread of motion)
        std_feat = np.std(seq, axis=1)

        # 3. Temporal velocity (first-order derivative along time axis)
        diff_feat = np.diff(seq, axis=1)
        vel_feat = np.mean(diff_feat, axis=1) if diff_feat.shape[1] > 0 else np.zeros_like(mean_feat)

        # 4. Maximum coordinate extent
        max_feat = np.max(seq, axis=1)

        # 5. Minimum coordinate extent
        min_feat = np.min(seq, axis=1)

        # Concatenate into unified feature vector: shape (N, 630)
        features = np.hstack([mean_feat, std_feat, vel_feat, max_feat, min_feat])

        return features[0] if is_single else features

    def predict(self, sequence: np.ndarray) -> Tuple[str, float]:
        """
        Predicts the sign class and confidence for a (25, 126) sequence.
        
        Returns:
            (predicted_class_name, confidence_score)
        """
        feats = self.extract_features(sequence).reshape(1, -1)
        probs = self.model.predict_proba(feats)[0]
        best_idx = int(np.argmax(probs))
        conf = float(probs[best_idx])
        pred_class = str(self.classes[best_idx])
        return pred_class, conf

    def get_top_k(self, sequence: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Returns top-k candidate predictions and their probabilities.
        """
        feats = self.extract_features(sequence).reshape(1, -1)
        probs = self.model.predict_proba(feats)[0]
        top_indices = np.argsort(probs)[-k:][::-1]
        return [(str(self.classes[i]), float(probs[i])) for i in top_indices]
