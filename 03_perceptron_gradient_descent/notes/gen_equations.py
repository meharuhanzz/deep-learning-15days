import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from _shared.mathfig import render

EQS = {
    "eq_perceptron": r"$\hat y = \sigma(w^\top x + b)$",
    "eq_boundary": r"$w^\top x + b = 0 \quad\text{(the decision boundary: a hyperplane)}$",
    "eq_minibatch_grad": r"$\nabla_w L \approx \dfrac{1}{|B|}\sum_{i \in B} \nabla_w L_i$",
}

if __name__ == "__main__":
    for name, latex in EQS.items():
        render(latex, f"{name}.png", fontsize=17)
        print("wrote", name)
