"""
Unit I: McCulloch-Pitts and Sigmoid Neuron Implementations from Scratch.
Aligned to CSAI3027P Deep Learning - Project 13: Sign Language Recognition.
"""

import numpy as np


class McCullochPittsNeuron:
    """
    McCulloch-Pitts Neuron (1943) - The foundational computational neuron model.
    Operates on binary inputs x_i in {0, 1} with excitatory and optional inhibitory inputs.
    Fires (output = 1) if weighted excitatory sum >= threshold AND no inhibitory input is active.
    
    In Sign Language Assistant:
    Acts as a real-time binary gesture gate (e.g. hand-presence / motion-trigger detection).
    """

    def __init__(self, weights: np.ndarray, threshold: float, inhibitory_indices: list = None):
        """
        Args:
            weights: 1D array of weights (positive values for excitatory).
            threshold: Activation threshold theta.
            inhibitory_indices: List of input indices that completely inhibit firing if active.
        """
        self.weights = np.asarray(weights, dtype=np.float64)
        self.threshold = float(threshold)
        self.inhibitory_indices = set(inhibitory_indices or [])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward evaluation.
        Args:
            x: shape (n_samples, n_features) or (n_features,)
        Returns:
            binary array of 0 or 1
        """
        x_arr = np.atleast_2d(x).astype(np.float64)
        n_samples = x_arr.shape[0]
        outputs = np.zeros(n_samples, dtype=np.int32)

        for i in range(n_samples):
            sample = x_arr[i]
            # Check inhibitory conditions
            inhibited = any(sample[idx] == 1 for idx in self.inhibitory_indices if idx < len(sample))
            if inhibited:
                outputs[i] = 0
                continue
            
            # Excitatory summation
            excitatory_mask = np.ones(len(sample), dtype=bool)
            for idx in self.inhibitory_indices:
                if idx < len(sample):
                    excitatory_mask[idx] = False

            net = np.dot(sample[excitatory_mask], self.weights[excitatory_mask])
            outputs[i] = 1 if net >= self.threshold else 0

        return outputs if x.ndim > 1 else outputs[0]


class SigmoidNeuron:
    """
    Single Sigmoid Neuron with continuous inputs and smooth differentiability.
    z = w^T x + b
    a = sigma(z) = 1 / (1 + exp(-z))
    da/dz = sigma(z) * (1 - sigma(z))
    """

    def __init__(self, n_features: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.1, size=(n_features, 1))
        self.b = 0.0
        self.last_x = None
        self.last_a = None
        self.last_z = None

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    @staticmethod
    def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        return a * (1.0 - a)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: shape (n_samples, n_features)
        Returns:
            a: shape (n_samples, 1)
        """
        self.last_x = np.asarray(x, dtype=np.float64)
        self.last_z = np.dot(self.last_x, self.w) + self.b
        self.last_a = self.sigmoid(self.last_z)
        return self.last_a

    def backward(self, d_loss_da: np.ndarray):
        """
        Computes analytical gradients for MSE or BCE loss.
        dL/dw = (1/m) X^T (dL/da * da/dz)
        dL/db = (1/m) sum(dL/da * da/dz)
        """
        m = self.last_x.shape[0]
        da_dz = self.sigmoid_derivative(self.last_a)
        dz = d_loss_da * da_dz  # (m, 1)

        dw = np.dot(self.last_x.T, dz) / m
        db = float(np.sum(dz) / m)
        return dw, db
