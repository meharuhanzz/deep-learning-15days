import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from _shared.mathfig import render

EQS = {
    "eq_dot": r"$w^\top x = \sum_i w_i x_i$",
    "eq_matmul": r"$(AB)_{ij} = \sum_k A_{ik} B_{kj}$",
    "eq_l2norm": r"$\|w\|_2 = \sqrt{\sum_i w_i^2}$",
    "eq_gradient": r"$\nabla f(x) = \left[\dfrac{\partial f}{\partial x_1}, \dots, \dfrac{\partial f}{\partial x_n}\right]^\top$",
    "eq_chain_scalar": r"$\dfrac{dy}{dx} = \dfrac{dy}{du}\cdot\dfrac{du}{dx}$",
    "eq_chain_backprop": r"$\dfrac{\partial L}{\partial w} = \dfrac{\partial L}{\partial \hat y}\cdot\dfrac{\partial \hat y}{\partial w}$",
    "eq_directional": r"$D_v f(x) = \nabla f(x)\cdot v = \|\nabla f\|\,\|v\|\cos\theta$",
    "eq_gaussian": r"$p(x) = \dfrac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\dfrac{(x-\mu)^2}{2\sigma^2}\right)$",
    "eq_bce": r"$\mathcal{L}(y,\hat y) = -\left[y\log \hat y + (1-y)\log(1-\hat y)\right]$",
    "eq_gd_update": r"$w \leftarrow w - \eta \,\nabla_w L$",
}

if __name__ == "__main__":
    for name, latex in EQS.items():
        render(latex, f"{name}.png", fontsize=18)
        print("wrote", name)
