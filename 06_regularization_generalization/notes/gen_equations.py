import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from _shared.mathfig import render

EQS = {
    "eq_l2reg": r"$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{data}} + \lambda \|w\|_2^2$",
    "eq_l1reg": r"$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{data}} + \lambda \|w\|_1$",
    "eq_dropout": r"$\tilde h = \dfrac{m \odot h}{1-p}, \quad m_i \sim \mathrm{Bernoulli}(1-p)$",
    "eq_batchnorm": r"$\hat x = \dfrac{x - \mu_B}{\sqrt{\sigma_B^2+\epsilon}}, \quad y = \gamma \hat x + \beta$",
    "eq_xavier": r"$\mathrm{Var}(w) = \dfrac{2}{n_{in}+n_{out}} \quad\text{(Xavier/Glorot, for tanh/sigmoid)}$",
    "eq_he": r"$\mathrm{Var}(w) = \dfrac{2}{n_{in}} \quad\text{(He/Kaiming, for ReLU)}$",
}

if __name__ == "__main__":
    for name, latex in EQS.items():
        render(latex, f"{name}.png", fontsize=16)
        print("wrote", name)
