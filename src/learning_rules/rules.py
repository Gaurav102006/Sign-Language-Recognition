"""
Unit III: Learning Rules Sandbox.
Hebbian, Modified Hebbian (Oja's Rule), Competitive Learning (Winner-Take-All),
Error-Correction (Delta Rule), and Energy-Based / Boltzmann Learning.
Aligned to CSAI3027P Deep Learning - Project 13: Sign Language Assistant.
"""

from typing import Dict, Tuple
import numpy as np


class HebbianLearning:
    """
    Classical Hebbian Rule (1949):
    'Neurons that fire together wire together'
    Delta W = eta * y * x^T
    Note: Without normalization, weights can grow unboundedly.
    """

    def __init__(self, n_features: int, n_neurons: int, lr: float = 0.01):
        self.lr = float(lr)
        self.weights = np.random.normal(0.0, 0.1, size=(n_neurons, n_features))

    def train_step(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: shape (n_features,)
        Returns:
            y: activation output
        """
        y = np.dot(self.weights, x)  # (n_neurons,)
        delta_w = self.lr * np.outer(y, x)
        self.weights += delta_w
        return y


class OjaModifiedHebbian:
    """
    Oja's Subspace Rule (1982) - Modified Hebbian Learning:
    Delta w = eta * y * (x - y * w) = eta * (y * x^T - y^2 * w)
    
    Self-stabilizing: Enforces ||w|| -> 1 and mathematically converges to the
    first principal component (eigenvector of the input covariance matrix).
    """

    def __init__(self, n_features: int, lr: float = 0.01):
        self.lr = float(lr)
        self.w = np.random.normal(0.0, 0.1, size=(n_features,))
        self.w /= np.linalg.norm(self.w) + 1e-15

    def train_step(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Args:
            x: shape (n_features,)
        Returns:
            (y, weight_norm)
        """
        y = float(np.dot(self.w, x))
        # Oja's update: prevents weight explosion without explicit re-normalization
        dw = self.lr * y * (x - y * self.w)
        self.w += dw
        weight_norm = float(np.linalg.norm(self.w))
        return y, weight_norm


class CompetitiveLearning:
    """
    Competitive Learning (Winner-Take-All / Unsupervised Cluster Neurons):
    Only the neuron with the highest inner product (or smallest Euclidean distance)
    to the input is permitted to adapt:
    winner k = argmin_i ||x - w_i||
    Delta w_k = eta * (x - w_k)
    """

    def __init__(self, n_features: int, n_clusters: int, lr: float = 0.05, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.n_clusters = n_clusters
        self.lr = float(lr)
        self.weights = rng.normal(0.0, 1.0, size=(n_clusters, n_features))
        # Normalize prototypes to unit hypersphere
        self.weights /= np.linalg.norm(self.weights, axis=1, keepdims=True) + 1e-15

    def train_step(self, x: np.ndarray) -> int:
        """
        Finds winner and updates prototype vector.
        """
        # Distance to each cluster prototype
        dists = np.linalg.norm(self.weights - x, axis=1)
        winner_idx = int(np.argmin(dists))

        # Update winner towards input
        self.weights[winner_idx] += self.lr * (x - self.weights[winner_idx])
        # Re-normalize
        self.weights[winner_idx] /= np.linalg.norm(self.weights[winner_idx]) + 1e-15
        return winner_idx


class ErrorCorrectionLearning:
    """
    Error-Correction Learning (Widrow-Hoff / Delta Rule / LMS):
    Updates weights in proportion to the difference between desired target and actual output:
    e = target - output
    Delta w = eta * e * x^T
    """

    def __init__(self, n_features: int, n_outputs: int, lr: float = 0.01):
        self.lr = float(lr)
        self.weights = np.zeros((n_outputs, n_features), dtype=np.float64)
        self.biases = np.zeros((n_outputs,), dtype=np.float64)

    def train_step(self, x: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Args:
            x: shape (n_features,)
            target: shape (n_outputs,)
        """
        y = np.dot(self.weights, x) + self.biases
        error = target - y
        self.weights += self.lr * np.outer(error, x)
        self.biases += self.lr * error
        mse = float(np.mean(error**2))
        return y, mse


class EnergyBasedBoltzmannDemo:
    """
    Restricted Boltzmann Machine / Energy-Based Learning Concept:
    Energy function: E(v, h) = - v^T W h - a^T v - b^T h
    Contrastive Divergence update (CD-1):
    Delta W = eta * ( <v * h^T>_data - <v * h^T>_model )
    
    Demonstrates generative association and energy minimization on sign gesture landmarks.
    """

    def __init__(self, n_visible: int, n_hidden: int, lr: float = 0.05, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = float(lr)
        self.w = rng.normal(0.0, 0.1, size=(n_visible, n_hidden))
        self.a = np.zeros(n_visible)  # visible bias
        self.b = np.zeros(n_hidden)   # hidden bias

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))

    def compute_energy(self, v: np.ndarray, h: np.ndarray) -> float:
        """E(v, h) = - v^T W h - a^T v - b^T h"""
        return float(-np.dot(v, np.dot(self.w, h)) - np.dot(self.a, v) - np.dot(self.b, h))

    def sample_h(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        prob_h = self._sigmoid(np.dot(v, self.w) + self.b)
        h_sample = (prob_h > np.random.rand(*prob_h.shape)).astype(np.float64)
        return prob_h, h_sample

    def sample_v(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        prob_v = self._sigmoid(np.dot(h, self.w.T) + self.a)
        v_sample = (prob_v > np.random.rand(*prob_v.shape)).astype(np.float64)
        return prob_v, v_sample

    def contrastive_divergence_step(self, v_data: np.ndarray) -> Dict:
        """
        One step of Contrastive Divergence (CD-1).
        """
        # Positive phase (data driven)
        prob_h0, h0 = self.sample_h(v_data)
        pos_grad = np.outer(v_data, prob_h0)

        # Negative phase (reconstruction / hallucination driven)
        prob_v1, v1 = self.sample_v(h0)
        prob_h1, h1 = self.sample_h(prob_v1)
        neg_grad = np.outer(prob_v1, prob_h1)

        # Contrastive update
        self.w += self.lr * (pos_grad - neg_grad)
        self.a += self.lr * (v_data - prob_v1)
        self.b += self.lr * (prob_h0 - prob_h1)

        recon_error = float(np.mean((v_data - prob_v1)**2))
        free_energy = float(-np.dot(self.a, v_data) - np.sum(np.log(1.0 + np.exp(np.dot(v_data, self.w) + self.b))))

        return {
            "reconstruction_error": recon_error,
            "free_energy": free_energy,
        }
