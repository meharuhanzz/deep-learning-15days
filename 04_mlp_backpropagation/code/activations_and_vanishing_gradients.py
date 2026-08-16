"""Day 4 hands-on, part 2:
  (a) the same MLP idea, now written the normal PyTorch way (nn.Module +
      autograd instead of hand-rolled arrays), to see the curved decision
      boundary a hidden layer buys you;
  (b) why sigmoid/tanh activations make DEEP networks hard to train:
      stack many layers and watch the gradient shrink toward the input,
      then show ReLU doesn't have the same problem.
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

from _shared.synthetic_data import make_moons

torch.manual_seed(0)


class TwoLayerMLP(nn.Module):
    """The standard way to define a model: subclass nn.Module, register
    sub-layers as attributes in __init__ (PyTorch finds them automatically
    for .parameters()), and describe the forward computation in forward()."""
    def __init__(self, n_in=2, n_hidden=8, n_out=1):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        h = self.act(self.fc1(x))
        return self.fc2(h)  # raw logits - loss fn applies sigmoid (as in Day 3)


def train_and_plot_boundary():
    X, y = make_moons(400, seed=0)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)

    model = TwoLayerMLP(n_hidden=8)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # Adam previewed, full story Day 5

    for epoch in range(300):
        logits = model(Xt).squeeze(-1)
        loss = loss_fn(logits, yt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        acc = ((torch.sigmoid(model(Xt).squeeze(-1)) > 0.5).float() == yt).float().mean().item()
    print(f"nn.Module MLP on moons: final acc {acc:.3f}")

    # decision boundary via a filled contour, since it's now curved and
    # can't be drawn as a single line the way Day 3's perceptron was
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
                          np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200))
    grid = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)
    with torch.no_grad():
        zz = torch.sigmoid(model(grid)).reshape(xx.shape).numpy()

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.contourf(xx, yy, zz, levels=20, cmap="coolwarm", alpha=0.6)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=12, edgecolor="k", linewidth=0.2)
    ax.set_title(f"MLP (1 hidden layer, tanh) — acc {acc:.2f}\ncurved boundary a straight line can't make")
    fig.tight_layout()
    fig.savefig("../notes/mlp_curved_boundary.png", dpi=130)
    plt.close(fig)
    print("wrote ../notes/mlp_curved_boundary.png")


def vanishing_gradient_demo():
    """Stack `depth` layers with a given activation, backprop a loss from
    the output all the way to the input, and record the gradient norm
    reaching each layer. Same random weights/input for every activation
    so the comparison isolates the activation function alone."""
    depth = 20
    width = 64
    torch.manual_seed(0)
    x0 = torch.randn(1, width)

    results = {}
    for name, act_cls in [("sigmoid", nn.Sigmoid), ("tanh", nn.Tanh), ("relu", nn.ReLU)]:
        torch.manual_seed(1)  # same init across activations
        layers = [nn.Linear(width, width) for _ in range(depth)]
        acts = [act_cls() for _ in range(depth)]

        x = x0.clone().requires_grad_(True)
        activations = [x]
        h = x
        for lin, act in zip(layers, acts):
            h = act(lin(h))
            h.retain_grad()  # keep .grad on intermediate tensors (normally freed)
            activations.append(h)
        loss = h.sum()
        loss.backward()

        grad_norms = [a.grad.norm().item() for a in activations if a.grad is not None]
        results[name] = grad_norms
        print(f"{name:8s}: grad norm at output layer {grad_norms[-1]:.4f}, "
              f"at layer closest to input {grad_norms[0]:.2e}")

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, norms in results.items():
        ax.plot(range(len(norms)), norms, marker="o", markersize=3, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("layer index (0 = input side, 20 = output side)")
    ax.set_ylabel("gradient norm reaching this layer (log scale)")
    ax.set_title(f"Vanishing gradients: {depth}-layer stack, same init/input")
    ax.legend()
    fig.tight_layout()
    fig.savefig("../notes/vanishing_gradients.png", dpi=130)
    plt.close(fig)
    print("wrote ../notes/vanishing_gradients.png")


if __name__ == "__main__":
    train_and_plot_boundary()
    print()
    vanishing_gradient_demo()
