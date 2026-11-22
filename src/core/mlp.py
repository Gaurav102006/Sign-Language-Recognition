"""
Unit I & II: Multilayer Feedforward Perceptron (MLP) from Mathematical Scratch.
Forward pass, loss computation, backpropagation, and parameter tracking.
Aligned to CSAI3027P Deep Learning - Project 13: Sign Language Assistant.
"""

from typing import List, Tuple, Dict
import numpy as np


class ScratchMLP:
    """
    Fully-connected Multilayer Perceptron implemented entirely in pure NumPy.
    Supports arbitrary layer sizes, ReLU / Sigmoid / Tanh activations,
    stable Softmax output, and exact backpropagation for multi-class classification.
    """

    def __init__(
        self,
        layer_dims: List[int],
        activation: str = "relu",
        weight_init: str = "he",
        seed: int = 42,
    ):
        """
        Args:
            layer_dims: e.g. [126, 64, 32, 98] for sign language keypoint classification.
            activation: 'relu', 'sigmoid', or 'tanh'.
            weight_init: 'he', 'xavier', 'random', or 'zeros'.
            seed: Reproducibility seed.
        """
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        self.activation = activation.lower()
        self.weight_init = weight_init.lower()
        self.seed = seed

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._init_weights()

        # Cache for backpropagation
        self.cache_a: List[np.ndarray] = []  # Activations (including input as a[0])
        self.cache_z: List[np.ndarray] = []  # Pre-activations

        # Gradients
        self.grad_weights: List[np.ndarray] = []
        self.grad_biases: List[np.ndarray] = []

    def _init_weights(self):
        rng = np.random.default_rng(self.seed)
        self.weights.clear()
        self.biases.clear()

        for l in range(self.num_layers):
            fan_in = self.layer_dims[l]
            fan_out = self.layer_dims[l + 1]

            if self.weight_init == "zeros":
                w = np.zeros((fan_in, fan_out), dtype=np.float64)
            elif self.weight_init == "random":
                w = rng.normal(0.0, 1.0, size=(fan_in, fan_out)).astype(np.float64)
            elif self.weight_init == "xavier":
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                w = rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float64)
            elif self.weight_init == "he":
                std = np.sqrt(2.0 / fan_in)
                w = rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float64)
            else:
                raise ValueError(f"Unknown weight initialization: {self.weight_init}")

            b = np.zeros((1, fan_out), dtype=np.float64)
            self.weights.append(w)
            self.biases.append(b)

    # Activation functions and their derivatives
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z_clip = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clip))

    @staticmethod
    def _sigmoid_deriv(a: np.ndarray) -> np.ndarray:
        return a * (1.0 - a)

    @staticmethod
    def _tanh(z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    @staticmethod
    def _tanh_deriv(a: np.ndarray) -> np.ndarray:
        return 1.0 - a**2

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_deriv(z: np.ndarray) -> np.ndarray:
        return (z > 0.0).astype(np.float64)

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax using max subtraction."""
        shift_z = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(shift_z)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through all layers.
        Args:
            x: shape (batch_size, input_dim)
        Returns:
            softmax probabilities of shape (batch_size, output_dim)
        """
        self.cache_a = [x.astype(np.float64)]
        self.cache_z = []

        a = x.astype(np.float64)
        for l in range(self.num_layers):
            w = self.weights[l]
            b = self.biases[l]
            z = np.dot(a, w) + b
            self.cache_z.append(z)

            # Hidden layers use chosen activation; final layer uses softmax
            if l < self.num_layers - 1:
                if self.activation == "relu":
                    a = self._relu(z)
                elif self.activation == "sigmoid":
                    a = self._sigmoid(z)
                elif self.activation == "tanh":
                    a = self._tanh(z)
                else:
                    raise ValueError(f"Unknown activation: {self.activation}")
            else:
                a = self.softmax(z)

            self.cache_a.append(a)

        return a

    def compute_loss(
        self, y_pred: np.ndarray, y_true: np.ndarray, l2_reg: float = 0.0
    ) -> float:
        """
        Multi-class Cross Entropy Loss:
        L = - (1/m) sum_{i=1}^m sum_{k=1}^K y_{ik} log(y_hat_{ik}) + (l2/2m) sum ||W||^2
        """
        m = y_true.shape[0]
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)

        # One-hot check
        if y_true.ndim == 1:
            loss = -np.sum(np.log(y_pred_clipped[np.arange(m), y_true.astype(int)])) / m
        else:
            loss = -np.sum(y_true * np.log(y_pred_clipped)) / m

        # Optional L2 regularization penalty
        if l2_reg > 0.0:
            reg_penalty = 0.5 * l2_reg * sum(np.sum(w**2) for w in self.weights) / m
            loss += reg_penalty

        return float(loss)

    def backward(
        self, y_pred: np.ndarray, y_true: np.ndarray, l2_reg: float = 0.0
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Analytical Backpropagation:
        For softmax + cross entropy output:
            dZ^[L] = y_pred - y_true
            dW^[l] = (1/m) (A^[l-1])^T dZ^[l] + (l2/m) W^[l]
            db^[l] = (1/m) sum(dZ^[l], axis=0, keepdims=True)
            dZ^[l-1] = (dZ^[l] (W^[l])^T) * g'(Z^[l-1])
        """
        m = y_true.shape[0]
        self.grad_weights = [None] * self.num_layers
        self.grad_biases = [None] * self.num_layers

        # Handle integer labels by converting to one-hot for gradient step
        if y_true.ndim == 1:
            y_one_hot = np.zeros_like(y_pred)
            y_one_hot[np.arange(m), y_true.astype(int)] = 1.0
        else:
            y_one_hot = y_true.astype(np.float64)

        # Output layer gradient
        dz = (y_pred - y_one_hot)  # (m, output_dim)

        for l in reversed(range(self.num_layers)):
            a_prev = self.cache_a[l]
            w = self.weights[l]

            dw = np.dot(a_prev.T, dz) / m
            if l2_reg > 0.0:
                dw += (l2_reg / m) * w
            db = np.sum(dz, axis=0, keepdims=True) / m

            self.grad_weights[l] = dw
            self.grad_biases[l] = db

            if l > 0:
                da_prev = np.dot(dz, w.T)
                z_prev = self.cache_z[l - 1]
                if self.activation == "relu":
                    dz = da_prev * self._relu_deriv(z_prev)
                elif self.activation == "sigmoid":
                    dz = da_prev * self._sigmoid_deriv(self.cache_a[l])
                elif self.activation == "tanh":
                    dz = da_prev * self._tanh_deriv(self.cache_a[l])

        return self.grad_weights, self.grad_biases

    def count_parameters(self) -> Dict[str, int]:
        total_w = sum(w.size for w in self.weights)
        total_b = sum(b.size for b in self.biases)
        return {"weights": total_w, "biases": total_b, "total": total_w + total_b}

    def get_params_flat(self) -> np.ndarray:
        """Flattens all weights and biases into a single 1D vector (for gradient checking)."""
        parts = []
        for w, b in zip(self.weights, self.biases):
            parts.append(w.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts)

    def set_params_flat(self, flat_params: np.ndarray):
        """Restores weights and biases from a 1D vector."""
        idx = 0
        for l in range(self.num_layers):
            w_shape = self.weights[l].shape
            w_size = self.weights[l].size
            self.weights[l] = flat_params[idx : idx + w_size].reshape(w_shape)
            idx += w_size

            b_shape = self.biases[l].shape
            b_size = self.biases[l].size
            self.biases[l] = flat_params[idx : idx + b_size].reshape(b_shape)
            idx += b_size

    def get_grads_flat(self) -> np.ndarray:
        """Flattens all calculated gradients into a single 1D vector."""
        parts = []
        for dw, db in zip(self.grad_weights, self.grad_biases):
            parts.append(dw.ravel())
            parts.append(db.ravel())
        return np.concatenate(parts)
