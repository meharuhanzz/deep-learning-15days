import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from _shared.mathfig import render

EQS = {
    "eq_softmax": r"$\mathrm{softmax}(z)_k = \dfrac{e^{z_k}}{\sum_j e^{z_j}}$",
    "eq_cce": r"$\mathcal{L} = -\sum_k y_k \log \hat y_k = -\log \hat y_{\,\text{true class}}$",
    "eq_mse": r"$\mathcal{L}_{\mathrm{MSE}} = \dfrac{1}{n}\sum_i (\hat y_i - y_i)^2$",
    "eq_mae": r"$\mathcal{L}_{\mathrm{MAE}} = \dfrac{1}{n}\sum_i |\hat y_i - y_i|$",
    "eq_momentum": r"$v \leftarrow \beta v + (1-\beta)\nabla_w L \qquad w \leftarrow w - \eta v$",
    "eq_adam1": r"$m \leftarrow \beta_1 m + (1-\beta_1)g \qquad v \leftarrow \beta_2 v + (1-\beta_2) g^2$",
    "eq_adam2": r"$\hat m = \dfrac{m}{1-\beta_1^t} \qquad \hat v = \dfrac{v}{1-\beta_2^t} \qquad w \leftarrow w - \eta\,\dfrac{\hat m}{\sqrt{\hat v}+\epsilon}$",
}

if __name__ == "__main__":
    for name, latex in EQS.items():
        render(latex, f"{name}.png", fontsize=16)
        print("wrote", name)
