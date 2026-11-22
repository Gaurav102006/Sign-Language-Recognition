"""
Unit Test: Unit III Learning Rules (Hebbian, Oja, Competitive, Error-Correction, Boltzmann).
"""

import unittest
import numpy as np
from src.learning_rules.rules import (
    HebbianLearning,
    OjaModifiedHebbian,
    CompetitiveLearning,
    ErrorCorrectionLearning,
    EnergyBasedBoltzmannDemo,
)


class TestLearningRules(unittest.TestCase):
    def test_hebbian(self):
        heb = HebbianLearning(n_features=5, n_neurons=3, lr=0.01)
        x = np.random.randn(5)
        y = heb.train_step(x)
        self.assertEqual(len(y), 3)

    def test_oja_self_stabilization(self):
        oja = OjaModifiedHebbian(n_features=5, lr=0.01)
        x = np.random.randn(5)
        for _ in range(50):
            _, norm = oja.train_step(x)
        # Oja's rule maintains normalized weight vector
        self.assertAlmostEqual(norm, 1.0, places=1)

    def test_competitive_learning(self):
        comp = CompetitiveLearning(n_features=6, n_clusters=3, lr=0.1)
        x = np.random.randn(6)
        winner = comp.train_step(x)
        self.assertIn(winner, [0, 1, 2])

    def test_error_correction(self):
        ec = ErrorCorrectionLearning(n_features=4, n_outputs=2, lr=0.05)
        x = np.array([1.0, -1.0, 0.5, 2.0])
        target = np.array([1.0, 0.0])
        # After training multiple steps on constant input, error should decrease
        _, initial_mse = ec.train_step(x, target)
        for _ in range(40):
            _, final_mse = ec.train_step(x, target)
        self.assertLess(final_mse, initial_mse)

    def test_boltzmann_cd(self):
        rbm = EnergyBasedBoltzmannDemo(n_visible=8, n_hidden=4, lr=0.05)
        v = np.random.binomial(1, 0.5, size=8).astype(np.float64)
        res = rbm.contrastive_divergence_step(v)
        self.assertIn("reconstruction_error", res)
        self.assertIn("free_energy", res)


if __name__ == "__main__":
    unittest.main()
