"""Day 13 hands-on, part 1: train a GAN on a 2D synthetic distribution
(Day 3's two-moons) instead of images - in 2D we can plot the ENTIRE
generated distribution against the real one directly, no dimensionality
reduction needed, making training dynamics fully visible.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _shared.synthetic_data import make_moons

torch.manual_seed(0)
Z_DIM = 8


class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, out_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, in_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),   # raw logit; BCEWithLogitsLoss applies sigmoid
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    X_real, _ = make_moons(2000, seed=0)
    X_real = torch.from_numpy(X_real)

    G = Generator()
    D = Discriminator()
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()

    snapshots = {}
    n_epochs = 3000
    batch_size = 128
    for epoch in range(n_epochs):
        idx = torch.randint(0, len(X_real), (batch_size,))
        real_batch = X_real[idx]
        z = torch.randn(batch_size, Z_DIM)
        fake_batch = G(z).detach()   # detach: don't backprop into G during D's update

        # --- D step: real -> label 1, fake -> label 0 (eq_gan_minimax) ---
        d_real = D(real_batch)
        d_fake = D(fake_batch)
        loss_D = loss_fn(d_real, torch.ones(batch_size)) + loss_fn(d_fake, torch.zeros(batch_size))
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        # --- G step: NON-SATURATING loss (eq_gan_nonsat) - maximize D(G(z))
        # directly via label-flipping, rather than minimizing log(1-D(G(z))),
        # which has a much weaker gradient early in training when D easily
        # rejects G's (still poor) samples.
        z = torch.randn(batch_size, Z_DIM)
        fake_batch = G(z)
        d_fake_for_g = D(fake_batch)
        loss_G = loss_fn(d_fake_for_g, torch.ones(batch_size))  # label-flip trick
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        if epoch % 500 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:4d}  loss_D {loss_D.item():.4f}  loss_G {loss_G.item():.4f}")
            with torch.no_grad():
                snapshots[epoch] = G(torch.randn(1000, Z_DIM)).numpy()

    fig, axes = plt.subplots(1, len(snapshots), figsize=(3.2 * len(snapshots), 3.2))
    X_real_np = X_real.numpy()
    for ax, (epoch, fake) in zip(axes, snapshots.items()):
        ax.scatter(X_real_np[:, 0], X_real_np[:, 1], s=4, alpha=0.3, color="gray", label="real")
        ax.scatter(fake[:, 0], fake[:, 1], s=4, alpha=0.5, color="#c0392b", label="generated")
        ax.set_title(f"epoch {epoch}", fontsize=9)
        ax.set_xlim(-2, 3); ax.set_ylim(-1.5, 2)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig("../notes/gan_training_progress.png", dpi=130)
    print("wrote ../notes/gan_training_progress.png")


if __name__ == "__main__":
    main()
