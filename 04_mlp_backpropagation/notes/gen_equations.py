import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from _shared.mathfig import render

EQS = {
    "eq_mlp_forward": r"$h = f(W_1 x + b_1) \qquad \hat y = \sigma(W_2 h + b_2)$",
    "eq_output_error": r"$\delta_2 = \dfrac{\partial L}{\partial z_2} = \hat y - y$",
    "eq_backprop_recursion": r"$\delta_1 = (W_2^\top \delta_2)\odot f'(z_1)$",
    "eq_weight_grad": r"$\dfrac{\partial L}{\partial W_l} = \delta_l\, h_{l-1}^\top \qquad \dfrac{\partial L}{\partial b_l} = \delta_l$",
    "eq_sigmoid_deriv": r"$\sigma'(z) = \sigma(z)\left(1-\sigma(z)\right)$",
    "eq_tanh_deriv": r"$\tanh'(z) = 1 - \tanh^2(z)$",
    "eq_relu": r"$\mathrm{ReLU}(z) = \max(0, z) \qquad \mathrm{ReLU}'(z) = \mathbb{1}[z>0]$",
}

if __name__ == "__main__":
    for name, latex in EQS.items():
        render(latex, f"{name}.png", fontsize=17)
        print("wrote", name)
