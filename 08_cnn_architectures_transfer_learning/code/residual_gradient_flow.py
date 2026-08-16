"""Day 8 hands-on, part 1: a THIRD, architectural fix to the vanishing-
gradient story running since Day 4 (activation choice) and Day 6 (weight
init). Here we hold BOTH of those fixed (same ReLU, same default init -
deliberately the WORSE setting from Day 6) and change only the
architecture: plain deep stack vs. the same stack with residual (skip)
connections added.
"""
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_plain_stack(depth=20, width=64, seed=1):
    torch.manual_seed(seed)
    x0 = torch.randn(1, width)
    torch.manual_seed(seed)
    layers = [nn.Linear(width, width) for _ in range(depth)]  # DEFAULT init, on purpose
    acts = [nn.ReLU() for _ in range(depth)]

    x = x0.clone().requires_grad_(True)
    activations = [x]
    h = x
    for lin, act in zip(layers, acts):
        h = act(lin(h))
        h.retain_grad()
        activations.append(h)
    h.sum().backward()
    return [a.grad.norm().item() for a in activations if a.grad is not None]


def run_residual_stack(depth=20, width=64, seed=1):
    """Identical layers/init/activation to run_plain_stack - the ONLY
    change is: h = x + ReLU(Linear(x))  instead of  h = ReLU(Linear(x))."""
    torch.manual_seed(seed)
    x0 = torch.randn(1, width)
    torch.manual_seed(seed)
    layers = [nn.Linear(width, width) for _ in range(depth)]  # same default init
    acts = [nn.ReLU() for _ in range(depth)]

    x = x0.clone().requires_grad_(True)
    activations = [x]
    h = x
    for lin, act in zip(layers, acts):
        h = h + act(lin(h))   # <-- the entire difference: an identity shortcut
        h.retain_grad()
        activations.append(h)
    h.sum().backward()
    return [a.grad.norm().item() for a in activations if a.grad is not None]


def main():
    plain = run_plain_stack()
    resid = run_residual_stack()
    print(f"plain    stack: grad norm at input-side layer = {plain[0]:.2e}")
    print(f"residual stack: grad norm at input-side layer = {resid[0]:.2e}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(range(len(plain)), plain, marker="o", markersize=3, color="#c0392b", label="plain (no shortcut)")
    ax.plot(range(len(resid)), resid, marker="o", markersize=3, color="#27ae60", label="residual (with shortcut)")
    ax.set_yscale("log")
    ax.set_xlabel("layer index (0 = input side, 20 = output side)")
    ax.set_ylabel("gradient norm reaching this layer (log scale)")
    ax.set_title("Same weights, same init, same activation — only +x shortcut differs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("../notes/residual_gradient_flow.png", dpi=130)
    print("wrote ../notes/residual_gradient_flow.png")


if __name__ == "__main__":
    main()
