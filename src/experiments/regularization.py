"""
Unit III: Regularization and Normalization Layers from Scratch.
Dropout, Batch Normalization, Layer Normalization, and Weight Penalties (L1/L2).
Aligned to CSAI3027P Deep Learning - Project 13: Sign Language Assistant.
"""

from typing import Tuple
import numpy as np


class ScratchDropout:
    """
    Inverted Dropout:
    During training: a_drop = (a * mask) / (1 - drop_rate)
    During testing: a_drop = a (no test-time scaling required)
    """

    def __init__(self, drop_rate: float = 0.5):
        self.drop_rate = float(drop_rate)
        self.mask = None
        self.training = True

    def forward(self, x: np.ndarray, seed: int = None) -> np.ndarray:
        if not self.training or self.drop_rate == 0.0:
            return x

        keep_prob = 1.0 - self.drop_rate
        if seed is not None:
            rng = np.random.default_rng(seed)
            self.mask = (rng.uniform(0.0, 1.0, size=x.shape) < keep_prob).astype(np.float64)
        else:
            self.mask = (np.random.rand(*x.shape) < keep_prob).astype(np.float64)

        return (x * self.mask) / keep_prob

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if not self.training or self.drop_rate == 0.0:
            return d_out
        keep_prob = 1.0 - self.drop_rate
        return (d_out * self.mask) / keep_prob


class ScratchBatchNorm1D:
    """
    Batch Normalization for 1D/2D feature matrices:
    During training:
      mu_B = (1/m) sum(x)
      var_B = (1/m) sum((x - mu_B)^2)
      x_hat = (x - mu_B) / sqrt(var_B + eps)
      y = gamma * x_hat + beta
      running_mean = momentum * running_mean + (1 - momentum) * mu_B
      running_var = momentum * running_var + (1 - momentum) * var_B
    During testing:
      x_hat = (x - running_mean) / sqrt(running_var + eps)
      y = gamma * x_hat + beta
    """

    def __init__(self, dim: int, momentum: float = 0.9, eps: float = 1e-5):
        self.dim = dim
        self.momentum = momentum
        self.eps = eps

        self.gamma = np.ones((1, dim), dtype=np.float64)
        self.beta = np.zeros((1, dim), dtype=np.float64)

        self.running_mean = np.zeros((1, dim), dtype=np.float64)
        self.running_var = np.ones((1, dim), dtype=np.float64)

        self.training = True
        self.cache_x_hat = None
        self.cache_var = None
        self.cache_x_mu = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            m = x.shape[0]
            mu = np.mean(x, axis=0, keepdims=True)
            x_mu = x - mu
            var = np.mean(x_mu**2, axis=0, keepdims=True)
            std_inv = 1.0 / np.sqrt(var + self.eps)
            x_hat = x_mu * std_inv

            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var

            self.cache_x_hat = x_hat
            self.cache_var = var
            self.cache_x_mu = x_mu
            self.cache_m = m

            return self.gamma * x_hat + self.beta
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * x_hat + self.beta

    def backward(self, d_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Exact analytical backward pass for Batch Normalization.
        """
        m = self.cache_m
        d_gamma = np.sum(d_out * self.cache_x_hat, axis=0, keepdims=True)
        d_beta = np.sum(d_out, axis=0, keepdims=True)

        std_inv = 1.0 / np.sqrt(self.cache_var + self.eps)
        dx_hat = d_out * self.gamma

        dvar = np.sum(dx_hat * self.cache_x_mu * -0.5 * (std_inv**3), axis=0, keepdims=True)
        dmu = np.sum(dx_hat * -std_inv, axis=0, keepdims=True) + dvar * np.mean(-2.0 * self.cache_x_mu, axis=0, keepdims=True)

        dx = dx_hat * std_inv + dvar * (2.0 * self.cache_x_mu / m) + dmu / m
        return dx, d_gamma, d_beta


class ScratchLayerNorm1D:
    """
    Layer Normalization:
    Computes mean and variance across feature dimensions independently for each sample.
    Crucial for sequence/LSTM and transformer models.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.gamma = np.ones((1, dim), dtype=np.float64)
        self.beta = np.zeros((1, dim), dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Normalize across feature axis (axis=-1)
        mu = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_hat = (x - mu) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta
