"""Day 8 hands-on, part 2: transfer learning. Load a ResNet18 pretrained
on ImageNet (1.28M real photos, 1000 classes), FREEZE its convolutional
feature extractor, and train only a new final layer for our 3-class
synthetic shape task. Compare against training the same-sized head on
RANDOM (untrained) ResNet18 features, to isolate how much of the benefit
is really "pretrained knowledge" vs. just "a reasonable fixed random
projection."

Real-world bridge this script has to handle explicitly: ImageNet models
expect 3-channel, ~224x224 input; our synthetic images are 1-channel,
32x32. This preprocessing gap is completely typical of applying a
pretrained model to a new domain, not specific to this toy example.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from _shared.synthetic_data import make_shape_images

torch.manual_seed(0)


def preprocess_for_resnet(x):
    """x: [N,1,32,32] in [-1,1] -> [N,3,224,224] roughly ImageNet-normalized.
    Two adaptations bridge the domain gap: replicate the single grayscale
    channel to 3 channels, and upsample spatially (ResNet's downsampling
    stages assume a much larger input than 32px)."""
    x = x.repeat(1, 3, 1, 1)                                  # 1 channel -> 3
    x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x + 1) / 2  # [-1,1] -> [0,1]
    return (x - mean) / std


def build_model(pretrained: bool, freeze_backbone: bool):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.resnet18(weights=weights)
    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
    backbone.fc = nn.Linear(backbone.fc.in_features, 3)  # new head, always trainable
    return backbone


def train_eval(model, X_train, y_train, X_val, y_val, epochs=5, lr=1e-3, batch_size=32):
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = len(X_train)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = preprocess_for_resnet(X_train[idx])
            logits = model(xb)
            loss = loss_fn(logits, y_train[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(preprocess_for_resnet(X_val))
            val_acc = (val_logits.argmax(-1) == y_val).float().mean().item()
        print(f"    epoch {epoch}  val_acc {val_acc:.3f}")
    return val_acc


def main():
    X, y = make_shape_images(600, size=32, seed=0)  # small: ResNet forward passes are expensive on CPU/tiny GPU budget
    X_train, y_train = X[:480], y[:480]
    X_val, y_val = X[480:], y[480:]

    print("=== (a) pretrained ResNet18, frozen backbone, train only the new head ===")
    model_pretrained = build_model(pretrained=True, freeze_backbone=True)
    acc_pretrained = train_eval(model_pretrained, X_train, y_train, X_val, y_val)

    print("\n=== (b) RANDOM (untrained) ResNet18, frozen backbone, same new head ===")
    model_random = build_model(pretrained=False, freeze_backbone=True)
    acc_random = train_eval(model_random, X_train, y_train, X_val, y_val)

    print(f"\nfinal: pretrained-frozen-backbone acc={acc_pretrained:.3f}  "
          f"random-frozen-backbone acc={acc_random:.3f}")
    print("(both only ever train the new 3-class head — the backbone's "
          "conv weights never update in either case)")


if __name__ == "__main__":
    main()
