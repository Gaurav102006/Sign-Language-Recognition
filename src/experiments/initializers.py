"""
Unit III: Weight Initialization Schemes and Variance Propagation Analysis.
Zero, Random, Xavier/Glorot, and He/Kaiming Initializers.
Aligned to CSAI3027P Deep Learning - Project 13: Sign Language Assistant.
"""

from typing import Dict, List
import numpy as np


class WeightInitializerSuite:
    """
    Demonstrates the impact of weight initialization schemes on signal variance
    and gradient propagation through deep networks.
    """

    @staticmethod
    def zero_init(fan_in: int, fan_out: int) -> np.ndarray:
        return np.zeros((fan_in, fan_out), dtype=np.float64)

    @staticmethod
    def random_normal_init(fan_in: int, fan_out: int, std: float = 1.0, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float64)

    @staticmethod
    def xavier_init(fan_in: int, fan_out: int, mode: str = "uniform", seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if mode == "uniform":
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float64)
        else:
            std = np.sqrt(2.0 / (fan_in + fan_out))
            return rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float64)

    @staticmethod
    def he_init(fan_in: int, fan_out: int, mode: str = "normal", seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if mode == "normal":
            std = np.sqrt(2.0 / fan_in)
            return rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float64)
        else:
            limit = np.sqrt(6.0 / fan_in)
            return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float64)

    @classmethod
    def test_variance_propagation(
        cls,
        n_layers: int = 10,
        dim: int = 126,
        batch_size: int = 64,
        activation: str = "relu",
    ) -> Dict[str, List[float]]:
        """
        Feeds unit-variance inputs through n_layers and measures the variance of activations
        at each layer for:
        - Small Random (std=0.01) -> vanishes to 0
        - Large Random (std=1.0) -> explodes or saturates
        - Xavier/Glorot -> stable for tanh
        - He/Kaiming -> stable for ReLU
        """
        rng = np.random.default_rng(42)
        x_init = rng.normal(0.0, 1.0, size=(batch_size, dim))

        schemes = ["Small_Random", "Large_Random", "Xavier", "He"]
        variances = {s: [] for s in schemes}

        for scheme in schemes:
            a = x_init.copy()
            for l in range(n_layers):
                if scheme == "Small_Random":
                    w = cls.random_normal_init(dim, dim, std=0.01, seed=42 + l)
                elif scheme == "Large_Random":
                    w = cls.random_normal_init(dim, dim, std=1.0, seed=42 + l)
                elif scheme == "Xavier":
                    w = cls.xavier_init(dim, dim, mode="normal", seed=42 + l)
                elif scheme == "He":
                    w = cls.he_init(dim, dim, mode="normal", seed=42 + l)

                z = np.dot(a, w)
                if activation == "relu":
                    a = np.maximum(0.0, z)
                elif activation == "tanh":
                    a = np.tanh(z)

                var = float(np.var(a))
                variances[scheme].append(var)

        return variances
