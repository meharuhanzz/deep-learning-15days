"""Day 12 hands-on: train a plain autoencoder AND a VAE on the synthetic
shapes dataset, both with a 2D bottleneck (so the latent space can be
plotted directly, no dimensionality reduction needed), and compare:
  (a) what the latent space looks like (plain AE vs VAE's KL-regularized
      space),
  (b) whether a random point in latent space decodes into something
      shape-like (only meaningful for VAE, by construction).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _shared.synthetic_data import make_shape_images

torch.manual_seed(0)
SIZE = 32
LATENT = 2  # deliberately 2D so we can plot the whole latent space directly


class Encoder(nn.Module):
    def __init__(self, latent_dim, variational):
        super().__init__()
        self.variational = variational
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),   # 32->16
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 16->8
            nn.Flatten(),
        )
        out_dim = 32 * 8 * 8
        if variational:
            self.fc_mu = nn.Linear(out_dim, latent_dim)
            self.fc_logvar = nn.Linear(out_dim, latent_dim)
        else:
            self.fc = nn.Linear(out_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        if self.variational:
            return self.fc_mu(h), self.fc_logvar(h)
        return self.fc(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 32 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),  # 8->16
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), nn.Tanh(),   # 16->32, match [-1,1] input range
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 32, 8, 8)
        return self.deconv(h)


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(LATENT, variational=False)
        self.decoder = Decoder(LATENT)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(LATENT, variational=True)
        self.decoder = Decoder(LATENT)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps                      # eq_reparam: the reparameterization trick
        return self.decoder(z), mu, logvar


def kl_divergence(mu, logvar):
    """eq_kl, summed over latent dims, averaged over the batch."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


def train_ae(X_train, epochs=15, lr=1e-3):
    model = AE()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        perm = torch.randperm(len(X_train))
        total_loss = 0.0
        for i in range(0, len(X_train), 64):
            xb = X_train[perm[i:i + 64]]
            recon, _ = model(xb)
            loss = nn.functional.mse_loss(recon, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(xb)
        print(f"  [AE]  epoch {epoch:2d}  recon_loss {total_loss/len(X_train):.4f}")
    return model


def train_vae(X_train, epochs=15, lr=1e-3, beta=0.001):
    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        perm = torch.randperm(len(X_train))
        total_recon, total_kl = 0.0, 0.0
        for i in range(0, len(X_train), 64):
            xb = X_train[perm[i:i + 64]]
            recon, mu, logvar = model(xb)
            recon_loss = nn.functional.mse_loss(recon, xb)
            kl = kl_divergence(mu, logvar)
            loss = recon_loss + beta * kl        # eq_elbo
            opt.zero_grad(); loss.backward(); opt.step()
            total_recon += recon_loss.item() * len(xb)
            total_kl += kl.item() * len(xb)
        print(f"  [VAE] epoch {epoch:2d}  recon_loss {total_recon/len(X_train):.4f}  "
              f"kl {total_kl/len(X_train):.4f}")
    return model


def main():
    X, y = make_shape_images(1500, size=SIZE, seed=0)

    print("=== training plain autoencoder (2D bottleneck) ===")
    ae = train_ae(X)
    print("\n=== training VAE (2D bottleneck) ===")
    vae = train_vae(X)

    # --- plot both latent spaces, colored by true shape class ---
    ae.eval(); vae.eval()
    with torch.no_grad():
        z_ae = ae.encoder(X).numpy()
        mu_vae, _ = vae.encoder(X)
        z_vae = mu_vae.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    names = ["circle", "square", "triangle"]
    for cls in range(3):
        mask = (y == cls).numpy()
        axes[0].scatter(z_ae[mask, 0], z_ae[mask, 1], s=8, label=names[cls])
        axes[1].scatter(z_vae[mask, 0], z_vae[mask, 1], s=8, label=names[cls])
    axes[0].set_title("Plain AE latent space")
    axes[1].set_title("VAE latent space (mu), KL-regularized toward N(0,1)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.set_xlabel("z[0]"); ax.set_ylabel("z[1]")
    fig.tight_layout()
    fig.savefig("../notes/latent_spaces.png", dpi=130)
    print("wrote ../notes/latent_spaces.png")

    # --- sample a grid of points from N(0,1) and decode with the VAE:
    # meaningful ONLY because training pushed the latent space toward this
    # exact distribution (the KL term) - doing this with the plain AE's
    # latent space has no such guarantee.
    grid = torch.linspace(-2.5, 2.5, 8)
    fig2, axes2 = plt.subplots(8, 8, figsize=(8, 8))
    with torch.no_grad():
        for i, gy in enumerate(grid):
            for j, gx in enumerate(grid):
                z = torch.tensor([[gx, gy]], dtype=torch.float32)
                img = vae.decoder(z)[0, 0].numpy()
                axes2[i, j].imshow(img, cmap="gray", vmin=-1, vmax=1)
                axes2[i, j].axis("off")
    fig2.suptitle("Decoding a regular grid sampled from the VAE's prior N(0,1)")
    fig2.tight_layout()
    fig2.savefig("../notes/vae_latent_grid.png", dpi=130)
    print("wrote ../notes/vae_latent_grid.png")


if __name__ == "__main__":
    main()
