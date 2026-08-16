import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from _shared.mathfig import render

EQS = {
    "eq_multivar_chain": r"$\dfrac{\partial L}{\partial x} = \sum_j \dfrac{\partial L}{\partial y_j}\cdot\dfrac{\partial y_j}{\partial x}$",
    "eq_grad_accum": r"$w.\mathrm{grad} \;+\!= \dfrac{\partial L}{\partial w}$",
}

if __name__ == "__main__":
    for name, latex in EQS.items():
        render(latex, f"{name}.png", fontsize=17)
        print("wrote", name)
