"""Day 6 hands-on, part 2: closes Day 4's vanishing-gradient cliffhanger.
Day 4 showed ReLU alone did NOT fix vanishing gradients in a naively-
initialized 20-layer stack (gradient norm shrank ~8 orders of magnitude
even with ReLU). Here we repeat the EXACT same experiment, changing only
the weight initialization scale from PyTorch's default to He/Kaiming init
(variance = 2/n_in, derived specifically for ReLU) - to see whether
initialization was really the missing piece.
"""
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run_stack(init_fn, depth=20, width=64, seed=1):
    torch.manual_seed(seed)
    x0 = torch.randn(1, width)
    torch.manual_seed(seed)
    layers = [nn.Linear(width, width) for _ in range(depth)]
    for lin in layers:
        init_fn(lin.weight)
        nn.init.zeros_(lin.bias)
    acts = [nn.ReLU() for _ in range(depth)]

    x = x0.clone().requires_grad_(True)
    activations = [x]
    h = x
    for lin, act in zip(layers, acts):
        h = act(lin(h))
        h.retain_grad()
        activations.append(h)
    loss = h.sum()
    loss.backward()
    return [a.grad.norm().item() for a in activations if a.grad is not None]


def default_init(w):
    pass  # leave nn.Linear's own default (uniform, ~1/sqrt(fan_in)) untouched


def he_init(w):
    nn.init.kaiming_normal_(w, nonlinearity="relu")  # Var(w) = 2/n_in


def main():
    results = {
        "default init (Day 4's result)": run_stack(default_init),
        "He/Kaiming init (this fix)": run_stack(he_init),
    }
    for name, norms in results.items():
        print(f"{name:32s}: grad norm at output {norms[-1]:.4f}, "
              f"at layer closest to input {norms[0]:.2e}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = {"default init (Day 4's result)": "#e67e22", "He/Kaiming init (this fix)": "#27ae60"}
    for name, norms in results.items():
        ax.plot(range(len(norms)), norms, marker="o", markersize=3,
                 color=colors[name], label=name)
    ax.set_yscale("log")
    ax.set_xlabel("layer index (0 = input side, 20 = output side)")
    ax.set_ylabel("gradient norm reaching this layer (log scale)")
    ax.set_title("Same ReLU stack as Day 4 — only the init scale changes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("../notes/init_scale_fix.png", dpi=130)
    print("wrote ../notes/init_scale_fix.png")


if __name__ == "__main__":
    main()
