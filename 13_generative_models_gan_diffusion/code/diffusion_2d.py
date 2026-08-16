"""Day 13 hands-on, part 2: a minimal DDPM-style diffusion model on the
same 2D two-moons distribution, for a direct, apples-to-apples comparison
with the GAN above - same data, same "generate new samples" goal,
completely different training principle (predict noise, not fool a
discriminator).
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
T = 100  # number of diffusion steps


def make_schedule(T, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


class DenoiseNet(nn.Module):
    """Predicts the noise epsilon added to x_t, given x_t and the
    timestep t (t is embedded as a simple scalar feature here - real
    diffusion models use a sinusoidal timestep embedding, the same idea
    as Day 10's positional encoding)."""
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(),   # input: x (2D) + t (1D, normalized)
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t_norm):
        inp = torch.cat([x, t_norm.unsqueeze(-1)], dim=-1)
        return self.net(inp)


def forward_diffuse(x0, t, alpha_bars):
    """eq_diffusion_forward: directly jump to any timestep t in closed form."""
    a_bar = alpha_bars[t].unsqueeze(-1)
    eps = torch.randn_like(x0)
    x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * eps
    return x_t, eps


def visualize_forward_process(X_real, alpha_bars):
    steps_to_show = [0, 10, 25, 50, 75, 99]
    fig, axes = plt.subplots(1, len(steps_to_show), figsize=(3 * len(steps_to_show), 3))
    for ax, t in zip(axes, steps_to_show):
        t_tensor = torch.full((len(X_real),), t, dtype=torch.long)
        x_t, _ = forward_diffuse(X_real, t_tensor, alpha_bars)
        ax.scatter(x_t[:, 0], x_t[:, 1], s=4, alpha=0.4, color="#2c3e50")
        ax.set_title(f"t={t}", fontsize=9)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    fig.suptitle("Forward process: two-moons gradually destroyed into pure noise")
    fig.tight_layout()
    fig.savefig("../notes/diffusion_forward.png", dpi=130)
    print("wrote ../notes/diffusion_forward.png")


def train_denoiser(X_real, alpha_bars, epochs=2000):
    model = DenoiseNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        idx = torch.randint(0, len(X_real), (256,))
        x0 = X_real[idx]
        t = torch.randint(0, T, (256,))
        x_t, eps = forward_diffuse(x0, t, alpha_bars)
        eps_pred = model(x_t, t.float() / T)
        loss = nn.functional.mse_loss(eps_pred, eps)   # eq_diffusion_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:4d}  noise_pred_loss {loss.item():.4f}")
    return model


@torch.no_grad()
def sample(model, betas, alphas, alpha_bars, n=1000):
    """Reverse process: start from pure noise, iteratively denoise one
    step at a time using the trained noise predictor - the DDPM ancestral
    sampler, in its simplest form."""
    x = torch.randn(n, 2)
    for t in reversed(range(T)):
        t_tensor = torch.full((n,), t, dtype=torch.long)
        eps_pred = model(x, t_tensor.float() / T)
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        beta_t = betas[t]
        mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_pred)
        if t > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * noise
        else:
            x = mean  # no noise added on the final step
    return x


def main():
    X_real, _ = make_moons(2000, seed=0)
    X_real = torch.from_numpy(X_real)
    betas, alphas, alpha_bars = make_schedule(T)

    visualize_forward_process(X_real, alpha_bars)

    print("\n=== training the denoising network ===")
    model = train_denoiser(X_real, alpha_bars)

    print("\n=== sampling: pure noise -> two-moons, via 100 reverse steps ===")
    generated = sample(model, betas, alphas, alpha_bars)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(X_real[:, 0], X_real[:, 1], s=4, alpha=0.3, color="gray", label="real")
    ax.scatter(generated[:, 0], generated[:, 1], s=4, alpha=0.5, color="#27ae60", label="diffusion-generated")
    ax.legend(fontsize=8)
    ax.set_title("Diffusion-generated samples vs. real data")
    fig.tight_layout()
    fig.savefig("../notes/diffusion_samples.png", dpi=130)
    print("wrote ../notes/diffusion_samples.png")


if __name__ == "__main__":
    main()
