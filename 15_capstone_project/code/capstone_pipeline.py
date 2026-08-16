"""Day 15 capstone: one complete, properly-engineered pipeline combining
techniques from across all 14 days:

  Day 3   - the training loop skeleton (forward/loss/zero_grad/backward/step)
  Day 6   - He initialization, BatchNorm, Dropout
  Day 7   - convolutional feature extraction
  Day 8   - residual connections
  Day 9   - (n/a - image task, not sequence)
  Day 14  - mixed precision (AMP), warmup+cosine LR schedule, grad clipping
  + a genuinely held-out TEST set, touched exactly once, at the very end
  + a Grad-CAM-style saliency check - "don't just trust the accuracy
    number, look at what the model is actually attending to."
"""
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _shared.synthetic_data import make_shape_images

torch.manual_seed(0)
SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ResBlock(nn.Module):
    """Day 8's residual connection + Day 6's BatchNorm, the standard
    modern combination."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)  # Day 8: the +x shortcut


class CapstoneCNN(nn.Module):
    def __init__(self, n_classes=3, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.block1 = ResBlock(32)
        self.down1 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(64), nn.ReLU())   # 32->16
        self.block2 = ResBlock(64)
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(128), nn.ReLU())  # 16->8
        self.block3 = ResBlock(128)
        self.gap = nn.AdaptiveAvgPool2d(1)   # global average pool: one number per channel
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(128, n_classes)
        self._init_weights()

    def _init_weights(self):
        # Day 6: He/Kaiming init, explicitly, for every conv layer
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_features=False):
        h = self.stem(x)
        h = self.block1(h)
        h = self.down1(h)
        h = self.block2(h)
        h = self.down2(h)
        features = self.block3(h)   # last conv feature map, used for Grad-CAM below
        pooled = self.gap(features).flatten(1)
        logits = self.head(self.dropout(pooled))
        return (logits, features) if return_features else logits


def warmup_cosine_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(progress * math.pi))


def train_capstone():
    print(f"device: {DEVICE}")
    X, y = make_shape_images(4000, size=SIZE, seed=0)
    y = y.long()

    # Day 6's evidence-discipline habit: a genuine 3-way split. Test set
    # is set aside now and not touched again until final evaluation.
    n = len(X)
    n_train, n_val = int(0.7 * n), int(0.15 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    print(f"train={len(X_train)}  val={len(X_val)}  test={len(X_test)} (untouched until the end)")

    model = CapstoneCNN().to(DEVICE)
    base_lr = 3e-3
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)  # Day 6: weight decay
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))   # Day 14: AMP

    epochs, batch_size = 20, 64
    steps_per_epoch = len(X_train) // batch_size
    total_steps = epochs * steps_per_epoch
    warmup_steps = steps_per_epoch * 2  # 2-epoch warmup

    history = {"train_loss": [], "val_acc": []}
    step = 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        for i in range(0, len(X_train) - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train[idx].to(DEVICE), y_train[idx].to(DEVICE)

            lr = warmup_cosine_lr(step, total_steps, warmup_steps, base_lr)   # Day 14
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad()
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE == "cuda")):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # Day 14: grad clipping
            scaler.step(opt)
            scaler.update()
            epoch_loss += loss.item()
            step += 1

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(DEVICE))
            val_acc = (val_logits.argmax(-1) == y_val.to(DEVICE)).float().mean().item()
        history["train_loss"].append(epoch_loss / steps_per_epoch)
        history["val_acc"].append(val_acc)
        print(f"  epoch {epoch:2d}  lr {lr:.5f}  train_loss {epoch_loss/steps_per_epoch:.4f}  val_acc {val_acc:.3f}")

    # --- the ONE evaluation on the held-out test set ---
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test.to(DEVICE))
        test_acc = (test_logits.argmax(-1) == y_test.to(DEVICE)).float().mean().item()
    print(f"\nFINAL HELD-OUT TEST ACCURACY: {test_acc:.4f}  (evaluated exactly once)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], color="#2c3e50")
    axes[0].set_title("train loss"); axes[0].set_xlabel("epoch")
    axes[1].plot(history["val_acc"], color="#27ae60")
    axes[1].axhline(test_acc, color="#c0392b", linestyle="--", label=f"final test acc {test_acc:.3f}")
    axes[1].set_title("validation accuracy per epoch"); axes[1].set_xlabel("epoch")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("../notes/capstone_training.png", dpi=130)
    print("wrote ../notes/capstone_training.png")

    return model, (X_test, y_test)


def grad_cam(model, x_single, target_class):
    """A simplified Grad-CAM: weight the last conv layer's feature maps by
    how much each channel's AVERAGE activation affects the target class's
    logit (the gradient of that logit w.r.t. each channel), then combine -
    a direct visual check of WHERE in the image the model's decision came
    from."""
    model.eval()
    x_single = x_single.unsqueeze(0).to(DEVICE).requires_grad_(False)
    logits, features = model(x_single, return_features=True)
    features.retain_grad()
    logits[0, target_class].backward()

    weights = features.grad[0].mean(dim=(1, 2))          # [channels] - avg gradient per channel
    cam = (weights[:, None, None] * features[0]).sum(0)   # weighted sum of feature maps
    cam = F.relu(cam)                                      # only positive contributions
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam[None, None], size=SIZE, mode="bilinear", align_corners=False)[0, 0]
    return cam.detach().cpu().numpy()


def interpretability_check(model, test_data):
    X_test, y_test = test_data
    names = ["circle", "square", "triangle"]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
    for col in range(4):
        img = X_test[col]
        true_label = y_test[col].item()
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(DEVICE)).argmax(-1).item()
        cam = grad_cam(model, img, target_class=pred)

        axes[0, col].imshow(img[0].numpy(), cmap="gray")
        axes[0, col].set_title(f"true={names[true_label]}\npred={names[pred]}", fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(img[0].numpy(), cmap="gray")
        axes[1, col].imshow(cam, cmap="jet", alpha=0.5)
        axes[1, col].set_title("Grad-CAM: where the decision came from", fontsize=8)
        axes[1, col].axis("off")
    fig.tight_layout()
    fig.savefig("../notes/capstone_gradcam.png", dpi=130)
    print("wrote ../notes/capstone_gradcam.png")


if __name__ == "__main__":
    model, test_data = train_capstone()
    print("\n=== interpretability check: does the model look at the shape? ===")
    interpretability_check(model, test_data)
