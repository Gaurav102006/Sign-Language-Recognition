"""
Unit Test: Unit V Sequence Models and VAE Generative Network.
"""

import unittest
import torch
from src.sequence_generative.sequence_models import (
    DeepRNNSignClassifier,
    BiLSTMSignClassifier,
    TransformerSignDecoder,
    GatingAblationExperiment,
)
from src.sequence_generative.vae_generative import SignLanguageVAE


class TestSequenceVAE(unittest.TestCase):
    def test_rnn_models(self):
        x = torch.randn(4, 25, 126)

        rnn = DeepRNNSignClassifier(in_dim=126, hidden_dim=64, num_classes=98)
        self.assertEqual(rnn(x).shape, (4, 98))

        lstm = BiLSTMSignClassifier(in_dim=126, hidden_dim=64, num_classes=98)
        self.assertEqual(lstm(x).shape, (4, 98))

        trans = TransformerSignDecoder(in_dim=126, d_model=64, nhead=4, num_classes=98)
        self.assertEqual(trans(x).shape, (4, 98))

    def test_gating_ablation(self):
        hist = GatingAblationExperiment.evaluate_cell_dynamics(n_steps=15)
        self.assertIn("sigmoid", hist)
        self.assertIn("tanh", hist)
        self.assertIn("step", hist)
        self.assertEqual(len(hist["sigmoid"]), 15)

    def test_vae_forward_and_loss(self):
        vae = SignLanguageVAE(seq_len=25, feat_dim=126, hidden_dim=64, latent_dim=8)
        x = torch.randn(4, 25, 126)
        recon_x, mu, logvar = vae(x)

        self.assertEqual(recon_x.shape, (4, 25, 126))
        self.assertEqual(mu.shape, (4, 8))
        self.assertEqual(logvar.shape, (4, 8))

        loss, r_loss, kl = vae.compute_loss(recon_x, x, mu, logvar)
        self.assertGreater(loss.item(), 0.0)

        # Sampling and interpolation
        samples = vae.sample_synthetic_gestures(num_samples=3)
        self.assertEqual(samples.shape, (3, 25, 126))

        interp = vae.interpolate_gestures(x[0], x[1], steps=5)
        self.assertEqual(interp.shape, (5, 25, 126))


if __name__ == "__main__":
    unittest.main()
