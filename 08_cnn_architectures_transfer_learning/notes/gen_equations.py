import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from _shared.mathfig import render

EQS = {
    "eq_residual": r"$y = F(x, W) + x$",
    "eq_residual_grad": r"$\dfrac{\partial y}{\partial x} = \dfrac{\partial F}{\partial x} + 1$",
}

if __name__ == "__main__":
    for name, latex in EQS.items():
        render(latex, f"{name}.png", fontsize=19)
        print("wrote", name)
