"""
Unit Test: Unit I & II Scratch MLP and Numerical Gradient Checking.
"""

import unittest
import numpy as np
from src.core.neuron import McCullochPittsNeuron, SigmoidNeuron
from src.core.mlp import ScratchMLP
from src.core.grad_check import GradientChecker
from src.core.optimizers import BatchGradientDescent, MomentumOptimizer, NesterovOptimizer


class TestMLPGradCheck(unittest.TestCase):
    def test_mcculloch_pitts(self):
        mp = McCullochPittsNeuron(weights=[1, 1], threshold=2)
        self.assertEqual(mp.forward(np.array([1, 1])), 1)
        self.assertEqual(mp.forward(np.array([1, 0])), 0)
        self.assertEqual(mp.forward(np.array([0, 1])), 0)

        # Inhibitory test
        mp_inh = McCullochPittsNeuron(weights=[1, 1], threshold=1, inhibitory_indices=[1])
        self.assertEqual(mp_inh.forward(np.array([1, 0])), 1)
        self.assertEqual(mp_inh.forward(np.array([1, 1])), 0)

    def test_sigmoid_neuron(self):
        sn = SigmoidNeuron(n_features=4)
        x = np.random.randn(5, 4)
        out = sn.forward(x)
        self.assertEqual(out.shape, (5, 1))
        dw, db = sn.backward(out - 0.5)
        self.assertEqual(dw.shape, (4, 1))

    def test_mlp_gradient_checking(self):
        """Verifies analytical backprop matches finite-difference numerical gradient."""
        mlp = ScratchMLP([8, 12, 4], activation="sigmoid", weight_init="xavier", seed=42)
        x = np.random.randn(6, 8)
        y = np.random.randint(0, 4, size=6)

        report = GradientChecker.check_model_gradients(mlp, x, y, epsilon=1e-5, max_params_to_check=25)
        self.assertTrue(report["passed"], f"Grad check failed: rel_error={report['relative_error']}")
        self.assertLess(report["relative_error"], 1e-5)

    def test_optimizers_step(self):
        mlp = ScratchMLP([8, 6, 3], activation="relu", seed=42)
        x = np.random.randn(4, 8)
        y = np.random.randint(0, 3, size=4)

        # SGD
        pred = mlp.forward(x)
        mlp.backward(pred, y)
        gd = BatchGradientDescent(lr=0.01)
        gd.step(mlp)

        # Momentum
        pred = mlp.forward(x)
        mlp.backward(pred, y)
        mom = MomentumOptimizer(lr=0.01, beta=0.9)
        mom.step(mlp)

        # Nesterov
        nag = NesterovOptimizer(lr=0.01, beta=0.9)
        nag.step_with_lookahead(mlp, x, y)


if __name__ == "__main__":
    unittest.main()
