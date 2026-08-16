"""Day 3 hands-on: train a single-layer perceptron with plain gradient
descent, on two different synthetic datasets, to build the training-loop
pattern you'll reuse for the rest of this course - and to see with your
own eyes WHY one layer isn't enough (setting up Day 4).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _shared.synthetic_data import make_linearly_separable, make_moons

torch.manual_seed(0)


def train(X, y, lr=0.5, epochs=200, log_every=20):
    """The canonical PyTorch training loop. Every model in this course,
    from a single neuron to a transformer, uses exactly this five-line
    skeleton: forward -> loss -> zero_grad -> backward -> step."""
    model = nn.Linear(2, 1)          # w in R^2, b in R  -> z = Wx + b
    loss_fn = nn.BCEWithLogitsLoss()  # sigmoid + binary cross-entropy, fused
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    history = []
    for epoch in range(epochs):
        logits = model(X).squeeze(-1)     # forward pass
        loss = loss_fn(logits, y)         # how wrong are we
        optimizer.zero_grad()             # clear old .grad (Day 2!)
        loss.backward()                   # chain rule, automatically
        optimizer.step()                  # w <- w - lr * w.grad  (Day 1's update rule)

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y).float().mean().item()
        history.append((loss.item(), acc))
        if epoch % log_every == 0:
            print(f"  epoch {epoch:3d}  loss {loss.item():.4f}  acc {acc:.3f}")
    return model, history


def plot_boundary(model, X, y, title, out_path):
    """The perceptron's decision boundary is exactly the line w^T x + b = 0
    (see notes.md eq_boundary) - we can draw it directly from the learned
    weights, not just infer it from scattered predictions."""
    w = model.weight.detach().numpy().flatten()
    b = model.bias.detach().item()

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    xs = X[:, 0].numpy(); ys = X[:, 1].numpy()
    ax.scatter(xs, ys, c=y.numpy(), cmap="coolwarm", s=12, edgecolor="k", linewidth=0.2)

    x_line = torch.linspace(xs.min() - 0.5, xs.max() + 0.5, 100)
    if abs(w[1]) > 1e-6:
        y_line = -(w[0] * x_line + b) / w[1]
        ax.plot(x_line, y_line, "k-", linewidth=2, label="w^Tx + b = 0")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    print("=== Case 1: linearly separable data (perceptron SHOULD succeed) ===")
    X_sep, y_sep = make_linearly_separable(400, seed=0)
    X_sep_t, y_sep_t = torch.from_numpy(X_sep), torch.from_numpy(y_sep)
    model_sep, hist_sep = train(X_sep_t, y_sep_t, lr=0.5, epochs=100)
    plot_boundary(model_sep, X_sep_t, y_sep_t,
                  f"Linearly separable — final acc {hist_sep[-1][1]:.2f}",
                  "../notes/boundary_linearly_separable.png")

    print("\n=== Case 2: two moons (perceptron SHOULD fail - no line can separate these) ===")
    X_moon, y_moon = make_moons(400, seed=0)
    X_moon_t, y_moon_t = torch.from_numpy(X_moon), torch.from_numpy(y_moon)
    model_moon, hist_moon = train(X_moon_t, y_moon_t, lr=0.5, epochs=100)
    plot_boundary(model_moon, X_moon_t, y_moon_t,
                  f"Two moons — final acc {hist_moon[-1][1]:.2f} (STUCK, see Day 4)",
                  "../notes/boundary_moons.png")

    print("\n=== Case 3: learning-rate sensitivity on the separable data ===")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for lr in [0.005, 0.1, 0.5, 3.0, 50.0]:
        torch.manual_seed(0)
        _, hist = train(X_sep_t, y_sep_t, lr=lr, epochs=60, log_every=1000)
        losses = [h[0] for h in hist]
        ax.plot(losses, label=f"lr={lr}")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_yscale("log")
    ax.set_title("Learning rate controls step size, not direction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("../notes/lr_sensitivity.png", dpi=130)
    plt.close(fig)
    print("  wrote lr_sensitivity.png")


if __name__ == "__main__":
    main()
