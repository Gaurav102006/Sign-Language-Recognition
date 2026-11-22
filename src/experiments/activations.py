"""
Unit III: Activation Functions and Vanishing Gradient / Saturation Analysis.
Step, Sigmoid, Tanh, ReLU, LeakyReLU, and Softmax implementations with exact derivatives.
Aligned to CSAI3027P Deep Learning - Project 13: Sign Language Assistant.
"""

from typing import Dict, List, Tuple
import numpy as np


class ActivationSuite:
    """
    Evaluates activation functions and their derivatives, demonstrating
    saturation zones, dead neuron regimes, and multi-layer gradient vanishing.
    """

    @staticmethod
    def step(x: np.ndarray) -> np.ndarray:
        return (x >= 0.0).astype(np.float64)

    @staticmethod
    def step_derivative(x: np.ndarray) -> np.ndarray:
        # Dirac delta approximation: 0 almost everywhere
        return np.zeros_like(x, dtype=np.float64)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        x_c = np.clip(x, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-x_c))

    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        s = ActivationSuite.sigmoid(x)
        return s * (1.0 - s)  # Max value is 0.25 at x=0

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        return 1.0 - np.tanh(x)**2  # Max value is 1.0 at x=0

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(np.float64)

    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0.0, x, alpha * x)

    @staticmethod
    def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0.0, 1.0, alpha)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        shift_x = x - np.max(x, axis=-1, keepdims=True)
        exps = np.exp(shift_x)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    @classmethod
    def simulate_deep_gradient_flow(
        cls, n_layers: int = 10, batch_size: int = 32, dim: int = 64
    ) -> Dict[str, List[float]]:
        """
        Simulates backpropagating a gradient of magnitude 1.0 through n_layers
        using Sigmoid, Tanh, and ReLU to quantitatively prove the Vanishing Gradient problem.
        """
        rng = np.random.default_rng(42)
        acts = ["sigmoid", "tanh", "relu", "leaky_relu"]
        grad_norms = {act: [] for act in acts}

        for act in acts:
            current_grad = np.ones((batch_size, dim), dtype=np.float64)
            # Standard random weights
            weights = [rng.normal(0.0, 1.0 / np.sqrt(dim), size=(dim, dim)) for _ in range(n_layers)]

            # Forward pre-activations
            a = rng.normal(0.0, 1.0, size=(batch_size, dim))
            zs = []
            for w in weights:
                z = np.dot(a, w)
                zs.append(z)
                if act == "sigmoid":
                    a = cls.sigmoid(z)
                elif act == "tanh":
                    a = cls.tanh(z)
                elif act == "relu":
                    a = cls.relu(z)
                elif act == "leaky_relu":
                    a = cls.leaky_relu(z)

            # Backward flow through layers
            grad = current_grad
            for l in reversed(range(n_layers)):
                z = zs[l]
                w = weights[l]
                if act == "sigmoid":
                    d_act = cls.sigmoid_derivative(z)
                elif act == "tanh":
                    d_act = cls.tanh_derivative(z)
                elif act == "relu":
                    d_act = cls.relu_derivative(z)
                elif act == "leaky_relu":
                    d_act = cls.leaky_relu_derivative(z)

                grad = np.dot(grad * d_act, w.T)
                norm = float(np.linalg.norm(grad) / (batch_size * dim))
                grad_norms[act].append(norm)

            grad_norms[act].reverse()  # Layer 1 to Layer N

        return grad_norms
