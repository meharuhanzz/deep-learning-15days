"""Day 14 hands-on:
  (a) measure REAL mixed-precision speedup on this machine's GPU (not a
      claim from a table - an actual timed comparison),
  (b) verify gradient accumulation produces (approximately) the same
      gradient as one large batch - continuing the course's verification
      habit (Days 1/2/4) one more time,
  (c) visualize a warmup+cosine learning-rate schedule against a constant
      LR.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _shared.synthetic_data import make_shape_images

torch.manual_seed(0)


def build_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(8),   # fixes the flattened size regardless of input resolution
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 3),
    )


def time_training(X, y, use_amp, device, steps=60, batch_size=64):
    model = build_cnn().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=use_amp)

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for step in range(steps):
        idx = torch.randint(0, len(X), (batch_size,))
        xb, yb = X[idx].to(device), y[idx].to(device)
        opt.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            logits = model(xb)
            loss = loss_fn(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    return elapsed


def amp_speed_comparison():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    X, y = make_shape_images(2000, size=64, seed=0)  # larger images -> AMP's advantage shows up more clearly
    y = y.long()

    # one untimed warmup run on each path - excludes CUDA context/cuDNN
    # autotuning startup cost from the timed measurement, which would
    # otherwise unfairly penalize whichever path runs first
    time_training(X, y, use_amp=False, device=device, steps=5)
    time_training(X, y, use_amp=True, device=device, steps=5)

    t_fp32 = time_training(X, y, use_amp=False, device=device, steps=100)
    t_amp = time_training(X, y, use_amp=True, device=device, steps=100)
    print(f"fp32 (full precision): {t_fp32:.3f}s for 100 steps")
    print(f"AMP  (mixed precision): {t_amp:.3f}s for 100 steps")
    if device == "cuda":
        print(f"speedup: {t_fp32/t_amp:.2f}x")
    else:
        print("(AMP's speedup is a GPU-tensor-core effect; on CPU this comparison "
              "is not meaningful - run on the GPU box to see the real number.)")


def gradient_accumulation_check():
    """One step on batch_size=64 vs. TWO accumulated micro-steps of
    batch_size=32 each (same 64 examples, same order) should produce
    (approximately) the same gradient - eq_grad_accum is an average, and
    summing two half-batches' gradients before dividing is the same
    arithmetic as one full-batch gradient, up to floating-point rounding."""
    torch.manual_seed(0)
    model_full = build_cnn()
    torch.manual_seed(0)
    model_accum = build_cnn()  # identical initialization

    X, y = make_shape_images(64, size=32, seed=1)
    y = y.long()
    loss_fn = nn.CrossEntropyLoss()

    # one full batch of 64
    logits = model_full(X)
    loss = loss_fn(logits, y)
    loss.backward()
    grad_full = model_full[0].weight.grad.clone()

    # two accumulated micro-batches of 32 (gradients naturally SUM across
    # backward() calls without an optimizer.step()/zero_grad() in between -
    # this IS gradient accumulation, no special API needed)
    for i in [0, 32]:
        logits = model_accum(X[i:i + 32])
        loss = loss_fn(logits, y[i:i + 32]) / 2   # divide by num micro-batches to match the full-batch AVERAGE
        loss.backward()
    grad_accum = model_accum[0].weight.grad.clone()

    max_diff = (grad_full - grad_accum).abs().max().item()
    print(f"full-batch grad vs. 2x accumulated half-batch grad: max diff = {max_diff:.2e}")
    assert max_diff < 1e-5, "gradient accumulation should match full-batch gradient closely"
    print("MATCH: accumulating gradients over micro-batches reproduces the full-batch gradient.")


def plot_lr_schedule():
    T, T_w, eta_max = 1000, 100, 1e-3
    import math
    etas_scheduled = []
    for t in range(T):
        if t < T_w:
            eta = eta_max * t / T_w
        else:
            eta = eta_max * 0.5 * (1 + math.cos((t - T_w) * math.pi / (T - T_w)))
        etas_scheduled.append(eta)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(etas_scheduled, color="#27ae60", label="warmup + cosine decay")
    ax.axhline(eta_max, color="#c0392b", linestyle="--", label="constant LR")
    ax.set_xlabel("training step"); ax.set_ylabel("learning rate")
    ax.set_title(f"Warmup ({T_w} steps) + cosine decay, vs. constant LR")
    ax.legend()
    fig.tight_layout()
    fig.savefig("../notes/lr_schedule.png", dpi=130)
    print("wrote ../notes/lr_schedule.png")


if __name__ == "__main__":
    print("=== (a) mixed precision speed comparison ===")
    amp_speed_comparison()
    print("\n=== (b) gradient accumulation equivalence check ===")
    gradient_accumulation_check()
    print("\n=== (c) learning-rate schedule ===")
    plot_lr_schedule()
