"""Day 6 hands-on, part 1: the cleanest possible demonstration of
overfitting - fit a high-capacity model to a SMALL noisy dataset with and
without L2 regularization (weight decay), and watch train/val loss
diverge in exactly the textbook shape.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _shared.synthetic_data import make_polynomial_regression

torch.manual_seed(0)


def make_big_mlp():
    """Deliberately oversized for this tiny dataset (60 points): 4 hidden
    layers of 128 units is enormous capacity relative to a cubic curve's
    true complexity - exactly the regime where overfitting shows up fast."""
    return nn.Sequential(
        nn.Linear(1, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 1),
    )


def train(weight_decay, epochs=2000):
    x_train, y_train = make_polynomial_regression(n=40, noise=0.5, seed=0)
    x_val, y_val = make_polynomial_regression(n=200, noise=0.5, seed=999)  # different seed = fresh noise draws
    xt, yt = torch.from_numpy(x_train), torch.from_numpy(y_train).unsqueeze(-1)
    xv, yv = torch.from_numpy(x_val), torch.from_numpy(y_val).unsqueeze(-1)

    torch.manual_seed(0)
    model = make_big_mlp()
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        pred = model(xt)
        loss = loss_fn(pred, yt)
        opt.zero_grad(); loss.backward(); opt.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yv).item()
        val_losses.append(val_loss)

    return model, train_losses, val_losses, (x_train, y_train, x_val, y_val)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for col, (wd, label) in enumerate([(0.0, "no regularization (weight_decay=0)"),
                                         (1e-2, "L2 regularization (weight_decay=1e-2)")]):
        model, tr, va, data = train(wd)
        x_train, y_train, x_val, y_val = data
        print(f"{label}: final train_loss={tr[-1]:.4f}  val_loss={va[-1]:.4f}  gap={va[-1]-tr[-1]:.4f}")

        ax_curve = axes[0, col]
        ax_curve.plot(tr, label="train", color="#2c3e50")
        ax_curve.plot(va, label="val", color="#c0392b")
        ax_curve.set_yscale("log")
        ax_curve.set_title(label, fontsize=9)
        ax_curve.set_xlabel("epoch"); ax_curve.set_ylabel("MSE loss")
        ax_curve.legend(fontsize=8)

        ax_fit = axes[1, col]
        x_grid = torch.linspace(-3, 3, 300).unsqueeze(-1)
        with torch.no_grad():
            y_grid = model(x_grid).squeeze(-1).numpy()
        y_true_grid = 0.5 * x_grid.squeeze(-1).numpy() ** 3 - x_grid.squeeze(-1).numpy() ** 2 - x_grid.squeeze(-1).numpy()
        ax_fit.scatter(x_train, y_train, s=14, color="#2c3e50", label="train points", zorder=3)
        ax_fit.plot(x_grid.squeeze(-1).numpy(), y_true_grid, "k--", linewidth=1, label="true function")
        ax_fit.plot(x_grid.squeeze(-1).numpy(), y_grid, color="#c0392b", linewidth=1.5, label="model fit")
        ax_fit.set_ylim(-15, 15)
        ax_fit.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig("../notes/overfitting_demo.png", dpi=130)
    print("wrote ../notes/overfitting_demo.png")


if __name__ == "__main__":
    main()
