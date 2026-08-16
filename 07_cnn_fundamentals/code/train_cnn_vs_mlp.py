"""Day 7 hands-on: train a small CNN AND a plain MLP on the identical
synthetic shape-classification task (circle/square/triangle, 32x32
grayscale), and compare not just accuracy but PARAMETER COUNT - the
concrete number behind "convolution is a better inductive bias for
images."
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _shared.synthetic_data import make_shape_images, to_loader

torch.manual_seed(0)
SIZE = 32


class SmallCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32->16
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16->8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class PlainMLP(nn.Module):
    """Same input, no convolution: every pixel connects to every hidden
    unit directly. Hidden width chosen so both models are trained for the
    same number of epochs on the same data - only the ARCHITECTURE differs."""
    def __init__(self, n_classes=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(SIZE * SIZE, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_eval(model, train_loader, val_X, val_y, epochs=8, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_acc = (model(val_X).argmax(-1) == val_y).float().mean().item()
        history.append(val_acc)
        print(f"    epoch {epoch}  val_acc {val_acc:.3f}")
    return history


def main():
    X, y = make_shape_images(3000, size=SIZE, seed=0)
    X_train, y_train = X[:2400], y[:2400]
    X_val, y_val = X[2400:], y[2400:]
    train_loader = to_loader(X_train, y_train, batch_size=32)

    cnn = SmallCNN()
    mlp = PlainMLP()
    print(f"CNN params: {count_params(cnn):,}")
    print(f"MLP params: {count_params(mlp):,}")

    print("\n=== training CNN ===")
    hist_cnn = train_eval(cnn, train_loader, X_val, y_val)
    print("\n=== training MLP (same data, same epochs, way more params) ===")
    hist_mlp = train_eval(mlp, train_loader, X_val, y_val)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist_cnn, "-o", label=f"CNN ({count_params(cnn):,} params)", color="#27ae60")
    ax.plot(hist_mlp, "-o", label=f"MLP ({count_params(mlp):,} params)", color="#c0392b")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation accuracy")
    ax.set_title("Same data, same training budget — CNN vs. plain MLP")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("../notes/cnn_vs_mlp.png", dpi=130)
    print("\nwrote ../notes/cnn_vs_mlp.png")

    # visualize what the first conv layer's filters look like after training
    filters = cnn.features[0].weight.detach().numpy()  # [8, 1, 3, 3]
    fig2, axes = plt.subplots(1, 8, figsize=(10, 1.5))
    for i, ax2 in enumerate(axes):
        ax2.imshow(filters[i, 0], cmap="gray")
        ax2.axis("off")
    fig2.suptitle("Learned 3x3 first-layer filters (8 of them)", fontsize=9)
    fig2.tight_layout()
    fig2.savefig("../notes/learned_filters.png", dpi=130)
    print("wrote ../notes/learned_filters.png")


if __name__ == "__main__":
    main()
