"""
Unit I: Optimization Algorithms Implemented from Scratch.
Batch GD, SGD, Momentum, and Nesterov Accelerated Gradient (NAG).
Aligned to CSAI3027P Deep Learning - Project 13: Sign Language Assistant.
"""

from typing import List
import numpy as np
from src.core.mlp import ScratchMLP


class BaseOptimizer:
    def __init__(self, lr: float = 0.01):
        self.lr = float(lr)

    def step(self, model: ScratchMLP):
        raise NotImplementedError


class BatchGradientDescent(BaseOptimizer):
    """
    Standard Batch Gradient Descent.
    Updates weights using the full dataset gradient:
    theta = theta - lr * grad
    """

    def step(self, model: ScratchMLP):
        for l in range(model.num_layers):
            model.weights[l] -= self.lr * model.grad_weights[l]
            model.biases[l] -= self.lr * model.grad_biases[l]


class SGDOptimizer(BaseOptimizer):
    """
    Stochastic Gradient Descent (with optional weight decay).
    """

    def __init__(self, lr: float = 0.01, weight_decay: float = 0.0):
        super().__init__(lr)
        self.weight_decay = weight_decay

    def step(self, model: ScratchMLP):
        for l in range(model.num_layers):
            gw = model.grad_weights[l]
            if self.weight_decay > 0.0:
                gw = gw + self.weight_decay * model.weights[l]
            model.weights[l] -= self.lr * gw
            model.biases[l] -= self.lr * model.grad_biases[l]


class MomentumOptimizer(BaseOptimizer):
    """
    Classical Polyak Momentum.
    v_t = beta * v_{t-1} + (1 - beta) * grad
    theta_t = theta_{t-1} - lr * v_t
    """

    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        super().__init__(lr)
        self.beta = beta
        self.v_w: List[np.ndarray] = []
        self.v_b: List[np.ndarray] = []

    def _init_velocity(self, model: ScratchMLP):
        if not self.v_w:
            self.v_w = [np.zeros_like(w) for w in model.weights]
            self.v_b = [np.zeros_like(b) for b in model.biases]

    def step(self, model: ScratchMLP):
        self._init_velocity(model)
        for l in range(model.num_layers):
            self.v_w[l] = self.beta * self.v_w[l] + (1.0 - self.beta) * model.grad_weights[l]
            self.v_b[l] = self.beta * self.v_b[l] + (1.0 - self.beta) * model.grad_biases[l]

            model.weights[l] -= self.lr * self.v_w[l]
            model.biases[l] -= self.lr * self.v_b[l]


class NesterovOptimizer(BaseOptimizer):
    """
    Nesterov Accelerated Gradient (NAG).
    Applies momentum step first to look ahead, then corrects using lookahead gradient:
    v_t = beta * v_{t-1} + lr * grad(theta - beta * v_{t-1})
    theta_t = theta_{t-1} - v_t
    """

    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        super().__init__(lr)
        self.beta = beta
        self.v_w: List[np.ndarray] = []
        self.v_b: List[np.ndarray] = []

    def _init_velocity(self, model: ScratchMLP):
        if not self.v_w:
            self.v_w = [np.zeros_like(w) for w in model.weights]
            self.v_b = [np.zeros_like(b) for b in model.biases]

    def step_with_lookahead(
        self, model: ScratchMLP, x: np.ndarray, y: np.ndarray, l2_reg: float = 0.0
    ):
        """
        NAG requires re-evaluating gradients at the lookahead parameters:
        theta_lookahead = theta - beta * v
        """
        self._init_velocity(model)

        # 1. Save original parameters
        orig_weights = [w.copy() for w in model.weights]
        orig_biases = [b.copy() for b in model.biases]

        # 2. Shift to lookahead point
        for l in range(model.num_layers):
            model.weights[l] -= self.beta * self.v_w[l]
            model.biases[l] -= self.beta * self.v_b[l]

        # 3. Compute gradient at lookahead
        y_pred = model.forward(x)
        model.backward(y_pred, y, l2_reg=l2_reg)

        # 4. Restore and perform update
        for l in range(model.num_layers):
            self.v_w[l] = self.beta * self.v_w[l] + self.lr * model.grad_weights[l]
            self.v_b[l] = self.beta * self.v_b[l] + self.lr * model.grad_biases[l]

            model.weights[l] = orig_weights[l] - self.v_w[l]
            model.biases[l] = orig_biases[l] - self.v_b[l]
