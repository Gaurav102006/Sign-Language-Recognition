"""
Unit V: Variational Autoencoder (VAE) for Sign Language Gesture Generation.
Reparameterization trick, KL-divergence loss, latent space visualization, and minority augmentation.
Aligned to CSAI3027P Deep Learning - Project 13: Sign Language Assistant.
"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SignLanguageVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for generative sign language modeling.
    Compresses (25 frames x 126 landmark features) into a continuous Gaussian latent manifold.
    Enables:
    1. Rare-sign data augmentation via latent prior sampling.
    2. Continuous interpolation between distinct gestures.
    3. 2D/3D latent space topological visualization.
    """

    def __init__(
        self,
        seq_len: int = 25,
        feat_dim: int = 126,
        hidden_dim: int = 256,
        latent_dim: int = 16,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.input_dim = seq_len * feat_dim
        self.latent_dim = latent_dim

        # Encoder Network: q_phi(z | x)
        self.encoder_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder Network: p_theta(x | z)
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, self.input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps input sequences to latent Gaussian distribution parameters (mu, log_var)."""
        x_flat = x.view(x.size(0), -1)
        h = self.encoder_net(x_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        The Reparameterization Trick:
        z = mu + std * epsilon, where epsilon ~ N(0, I)
        Bypasses stochasticity so gradients backpropagate smoothly through mu and sigma.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs sequence from latent code."""
        recon_flat = self.decoder_net(z)
        return recon_flat.view(-1, self.seq_len, self.feat_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def compute_loss(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta_kl: float = 0.001,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Evidence Lower Bound (ELBO) Loss:
        L_VAE = Reconstruction_Loss + beta * D_KL( q(z|x) || p(z) )
        """
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta_kl * kl_div

        return total_loss, float(recon_loss.item()), float(kl_div.item())

    @torch.no_grad()
    def sample_synthetic_gestures(self, num_samples: int = 10) -> np.ndarray:
        """Samples synthetic sign gestures from standard Gaussian prior z ~ N(0, I)."""
        self.eval()
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        synthetic = self.decode(z)
        return synthetic.cpu().numpy()

    @torch.no_grad()
    def interpolate_gestures(
        self, seq_a: torch.Tensor, seq_b: torch.Tensor, steps: int = 10
    ) -> np.ndarray:
        """
        Smoothly interpolates between two sign gestures in the continuous latent manifold.
        """
        self.eval()
        mu_a, _ = self.encode(seq_a.unsqueeze(0))
        mu_b, _ = self.encode(seq_b.unsqueeze(0))

        alphas = np.linspace(0.0, 1.0, steps)
        interpolated = []

        for alpha in alphas:
            z_interp = (1.0 - alpha) * mu_a + alpha * mu_b
            decoded = self.decode(z_interp).squeeze(0)
            interpolated.append(decoded.cpu().numpy())

        return np.stack(interpolated, axis=0)
