"""
Data Loader and Partitioning Suite for Sign Language Recognition.
Produces leakage-safe Train, Validation, and Test partitions with label encoding.
Aligned to CSAI3027P Deep Learning - Project 13: Sign Language Assistant.
"""

import os
from typing import Dict, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


class SignDatasetManager:
    """
    Manages loading, label encoding, and leakage-safe partitioning of the
    4,740-sample 98-class Sign Language landmark dataset.
    """

    def __init__(self, data_dir: str = "/Users/gauravgoyal/Desktop/Sign_Language"):
        self.data_dir = data_dir
        self.x_path = os.path.join(data_dir, "X_train_ready.npy")
        self.y_path = os.path.join(data_dir, "y_train_ready.npy")
        self.classes_path = os.path.join(data_dir, "classes.npy")
        self.encoder_path = os.path.join(data_dir, "label_encoder.pkl")

        self.label_encoder = None
        self.classes = None

    def load_raw_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Loads numpy arrays directly from disk."""
        if not os.path.exists(self.x_path) or not os.path.exists(self.y_path):
            raise FileNotFoundError(f"Missing dataset files in {self.data_dir}")

        x = np.load(self.x_path, allow_pickle=True).astype(np.float32)
        y = np.load(self.y_path, allow_pickle=True)

        if os.path.exists(self.classes_path):
            self.classes = np.load(self.classes_path, allow_pickle=True)
        else:
            self.classes = np.unique(y)

        if os.path.exists(self.encoder_path):
            self.label_encoder = joblib.load(self.encoder_path)
        else:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.classes)

        return x, y, self.classes

    def get_partitions(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        seed: int = 42,
    ) -> Dict[str, np.ndarray]:
        """
        Creates strictly separated, stratified Train, Validation, and Test splits.
        Default: 70% Train, 15% Validation, 15% Test.
        """
        x, y, _ = self.load_raw_data()
        y_int = self.label_encoder.transform(y)

        # First split into train_val and test
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            x, y_int, test_size=test_size, random_state=seed, stratify=y_int
        )

        # Next split train_val into train and val
        relative_val_size = val_size / (1.0 - test_size)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val, test_size=relative_val_size, random_state=seed, stratify=y_train_val
        )

        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test,
            "classes": self.classes,
            "n_classes": len(self.classes),
            "label_encoder": self.label_encoder,
        }

    def get_dataset_metadata(self) -> Dict:
        """Inspects and returns dataset governance and summary statistics."""
        x, y, classes = self.load_raw_data()
        unique, counts = np.unique(y, return_counts=True)
        return {
            "total_samples": len(x),
            "sequence_length": x.shape[1],
            "feature_dim": x.shape[2],
            "num_classes": len(classes),
            "min_samples_per_class": int(np.min(counts)),
            "max_samples_per_class": int(np.max(counts)),
            "mean_samples_per_class": float(np.mean(counts)),
            "class_names_sample": classes[:10].tolist(),
        }
