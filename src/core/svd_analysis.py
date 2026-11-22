"""
Unit I: Singular Value Decomposition (SVD) for Feature Analysis, Compression, and Diagnostics.
Applied directly to Sign Language hand landmark coordinates and motion trajectories.
Aligned to CSAI3027P Deep Learning - Project 13.
"""

from typing import Dict, Tuple
import numpy as np


class LandmarkSVDAnalyzer:
    """
    Performs Singular Value Decomposition (SVD) on gesture landmark matrices:
    X = U Sigma V^T
    
    Provides:
    1. Singular value spectrum and cumulative energy distribution.
    2. Low-rank compression of 126-dimensional landmark frames.
    3. Reconstruction error (relative Frobenius norm).
    4. Condition number kappa = sigma_max / sigma_min for numerical stability diagnostics.
    """

    def __init__(self, x: np.ndarray):
        """
        Args:
            x: 2D array of shape (n_samples, n_features) or reshaped (n_sequences * n_frames, 126)
        """
        if x.ndim > 2:
            self.x = x.reshape(-1, x.shape[-1]).astype(np.float64)
        else:
            self.x = np.asarray(x, dtype=np.float64)

        # Center data
        self.mean = np.mean(self.x, axis=0)
        self.x_centered = self.x - self.mean

        # Compute full SVD
        self.u, self.s, self.vt = np.linalg.svd(self.x_centered, full_matrices=False)
        self.total_variance = np.sum(self.s**2)
        self.cumulative_energy = np.cumsum(self.s**2) / (self.total_variance + 1e-15)

    def get_singular_values(self) -> np.ndarray:
        return self.s

    def get_cumulative_energy(self) -> np.ndarray:
        return self.cumulative_energy

    def get_condition_number(self) -> float:
        """
        Diagnostics metric: kappa = sigma_max / sigma_min.
        High condition number indicates collinearity/redundancy in hand landmark tracking.
        """
        if len(self.s) == 0 or self.s[-1] == 0:
            return float("inf")
        return float(self.s[0] / (self.s[-1] + 1e-15))

    def compress(self, k: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compresses landmarks to top-k principal components.
        
        Returns:
            x_compressed: Low-rank reconstructed landmarks (same shape as original)
            z_lowdim: Latent representation of shape (n_samples, k)
            relative_error: Relative Frobenius norm error ||X - X_k||_F / ||X||_F
        """
        k = min(k, len(self.s))
        uk = self.u[:, :k]
        sk = np.diag(self.s[:k])
        vtk = self.vt[:k, :]

        # Low-dimensional embedding
        z_lowdim = np.dot(self.x_centered, vtk.T)

        # Reconstructed matrix
        x_recon_centered = np.dot(uk, np.dot(sk, vtk))
        x_reconstructed = x_recon_centered + self.mean

        # Reconstruction error
        norm_orig = np.linalg.norm(self.x_centered, "fro")
        norm_diff = np.linalg.norm(self.x_centered - x_recon_centered, "fro")
        relative_error = float(norm_diff / (norm_orig + 1e-15))

        return x_reconstructed, z_lowdim, relative_error

    def analyze(self, k_components: list = [4, 8, 16, 32, 64]) -> Dict:
        """Runs a complete diagnostic report across multiple compression ranks."""
        results = {
            "n_samples": self.x.shape[0],
            "n_features": self.x.shape[1],
            "condition_number": self.get_condition_number(),
            "top_10_singular_values": self.s[:10].tolist(),
            "compression_benchmarks": [],
        }

        for k in k_components:
            if k <= len(self.s):
                _, _, rel_err = self.compress(k)
                energy = float(self.cumulative_energy[k - 1])
                results["compression_benchmarks"].append({
                    "k": k,
                    "variance_retained_pct": round(energy * 100.0, 2),
                    "relative_frobenius_error": round(rel_err, 4),
                    "compression_ratio": round(self.x.shape[1] / k, 2),
                })

        return results
