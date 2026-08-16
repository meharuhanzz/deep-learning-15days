import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from _shared.mathfig import render

EQS = {
    "eq_cls_head": r"$\hat y = \mathrm{softmax}(W \cdot h_{[\mathrm{CLS}]} + b)$",
}

if __name__ == "__main__":
    for name, latex in EQS.items():
        render(latex, f"{name}.png", fontsize=18)
        print("wrote", name)
