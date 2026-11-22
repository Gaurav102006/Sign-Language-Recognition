"""
Unit III: Data Preparation and Augmentation for Sign Language Hand Landmarks.
Spatial scaling, rotation, translation jitter, Gaussian noise, frame dropout, and mirroring.
Aligned to CSAI3027P Deep Learning - Project 13: Sign Language Assistant.
"""

from typing import Tuple
import numpy as np


class LandmarkAugmenter:
    """
    Realistic kinematic data augmentation pipeline specifically calibrated for
    3D Sign Language hand landmarks (2 hands x 21 joints x (x,y,z) = 126 dims per frame).
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def add_gaussian_noise(self, seq: np.ndarray, std: float = 0.015) -> np.ndarray:
        """Simulates camera sensor noise and keypoint jitter."""
        noise = self.rng.normal(0.0, std, size=seq.shape)
        return seq + noise

    def random_scale(self, seq: np.ndarray, scale_range: Tuple[float, float] = (0.85, 1.15)) -> np.ndarray:
        """Simulates signer standing closer or farther from the camera."""
        scale = self.rng.uniform(scale_range[0], scale_range[1])
        return seq * scale

    def random_translation(self, seq: np.ndarray, shift_std: float = 0.03) -> np.ndarray:
        """Simulates signer moving horizontally or vertically in frame."""
        shift = self.rng.normal(0.0, shift_std, size=(1, seq.shape[-1]))
        return seq + shift

    def random_rotation_2d(self, seq: np.ndarray, max_angle_deg: float = 15.0) -> np.ndarray:
        """
        Rotates (x, y) coordinates around origin (wrist) in the 2D plane by angle theta.
        """
        angle_rad = np.radians(self.rng.uniform(-max_angle_deg, max_angle_deg))
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        r_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        seq_rot = seq.copy()
        # seq has shape (T, 126) -> 42 landmarks with (x, y, z)
        t, total_coords = seq.shape
        n_landmarks = total_coords // 3

        for frame_idx in range(t):
            coords = seq_rot[frame_idx].reshape(n_landmarks, 3)
            # Rotate (x, y) coordinates
            xy = coords[:, :2]
            rotated_xy = np.dot(xy, r_matrix.T)
            coords[:, :2] = rotated_xy
            seq_rot[frame_idx] = coords.reshape(-1)

        return seq_rot

    def random_frame_dropout(self, seq: np.ndarray, max_drop: int = 3) -> np.ndarray:
        """
        Simulates intermittent camera frame drops by masking and linear interpolation.
        """
        seq_aug = seq.copy()
        t = len(seq)
        n_drop = self.rng.integers(1, max_drop + 1)
        drop_indices = self.rng.choice(np.arange(1, t - 1), size=n_drop, replace=False)

        for idx in drop_indices:
            # Linear interpolation between adjacent frames
            seq_aug[idx] = 0.5 * (seq_aug[idx - 1] + seq_aug[idx + 1])

        return seq_aug

    def augment_pipeline(self, seq: np.ndarray) -> np.ndarray:
        """Applies a randomized composition of augmentations."""
        out = seq.copy()
        if self.rng.random() > 0.5:
            out = self.add_gaussian_noise(out)
        if self.rng.random() > 0.5:
            out = self.random_scale(out)
        if self.rng.random() > 0.5:
            out = self.random_translation(out)
        if self.rng.random() > 0.5:
            out = self.random_rotation_2d(out)
        if self.rng.random() > 0.7:
            out = self.random_frame_dropout(out)
        return out
