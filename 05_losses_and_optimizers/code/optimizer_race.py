"""Day 5 hands-on: watch SGD, SGD+momentum, and Adam take completely
different PATHS to the same minimum, on a deliberately ill-conditioned
loss surface (steep in one direction, shallow in another - like a narrow
valley). This single picture explains why momentum and adaptive learning
rates were invented: plain SGD oscillates across the steep direction while
crawling along the shallow one.

Loss surface: L(x, y) = 0.05*x^2 + 5*y^2  (steep in y, shallow in x)
Global minimum at (0, 0).
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def loss_fn(p):
    x, y = p[0], p[1]
    return 0.05 * x ** 2 + 5.0 * y ** 2


def run(optimizer_name, start=(-4.0, 1.0), steps=60, **opt_kwargs):
    p = torch.tensor(start, requires_grad=True)
    if optimizer_name == "sgd":
        opt = torch.optim.SGD([p], lr=opt_kwargs.get("lr", 0.19))
    elif optimizer_name == "sgd_momentum":
        opt = torch.optim.SGD([p], lr=opt_kwargs.get("lr", 0.19), momentum=0.9)
    elif optimizer_name == "adam":
        opt = torch.optim.Adam([p], lr=opt_kwargs.get("lr", 0.3))
    else:
        raise ValueError(optimizer_name)

    path = [p.detach().numpy().copy()]
    for _ in range(steps):
        opt.zero_grad()
        L = loss_fn(p)
        L.backward()
        opt.step()
        path.append(p.detach().numpy().copy())
    return np.array(path)


def main():
    paths = {name: run(name) for name in ["sgd", "sgd_momentum", "adam"]}
    for name, path in paths.items():
        final = path[-1]
        final_loss = 0.05 * final[0] ** 2 + 5.0 * final[1] ** 2
        print(f"{name:14s} final (x,y)=({final[0]: .4f},{final[1]: .4f})  loss={final_loss:.6f}")

    xx, yy = np.meshgrid(np.linspace(-5, 2, 200), np.linspace(-2, 2, 200))
    zz = 0.05 * xx ** 2 + 5.0 * yy ** 2

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contour(xx, yy, zz, levels=25, cmap="Greys", linewidths=0.6)
    colors = {"sgd": "#c0392b", "sgd_momentum": "#e67e22", "adam": "#27ae60"}
    for name, path in paths.items():
        ax.plot(path[:, 0], path[:, 1], "-o", markersize=2.5, linewidth=1,
                 color=colors[name], label=name)
    ax.plot(0, 0, "k*", markersize=14, label="minimum")
    ax.set_xlabel("x (shallow direction)")
    ax.set_ylabel("y (steep direction)")
    ax.set_title("Same start, same step budget, same base LR family — different paths")
    ax.legend()
    fig.tight_layout()
    fig.savefig("../notes/optimizer_paths.png", dpi=130)
    print("wrote ../notes/optimizer_paths.png")


if __name__ == "__main__":
    main()
