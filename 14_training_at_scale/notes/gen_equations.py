import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from _shared.mathfig import render

EQS = {
    "eq_grad_clip": r"$g \leftarrow g \cdot \min\!\left(1,\ \dfrac{\mathrm{clip\_norm}}{\|g\|}\right)$",
    "eq_grad_accum": r"$\nabla_w L \approx \dfrac{1}{k}\sum_{i=1}^{k} \nabla_w L_i \quad\text{(k micro-batches, one step)}$",
    "eq_warmup": r"$\eta(t) = \eta_{max}\cdot t / T_w \qquad (t < T_w,\ \text{linear warmup})$",
    "eq_cosine": r"$\eta(t) = \eta_{max}\cdot\frac{1}{2}\!\left(1+\cos\frac{(t-T_w)\pi}{T-T_w}\right) \qquad (t \geq T_w)$",
}

if __name__ == "__main__":
    for name, latex in EQS.items():
        render(latex, f"{name}.png", fontsize=15)
        print("wrote", name)
