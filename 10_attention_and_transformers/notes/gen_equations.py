import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from _shared.mathfig import render

EQS = {
    "eq_attention": r"$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}\right)V$",
    "eq_scaling_variance": r"$\mathrm{Var}(q\cdot k) = d_k \quad\Rightarrow\quad \mathrm{Var}\!\left(\dfrac{q\cdot k}{\sqrt{d_k}}\right)=1$",
    "eq_multihead": r"$\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\,W^O$",
    "eq_posenc": r"$PE_{(pos,2i)}=\sin\!\left(\dfrac{pos}{10000^{2i/d}}\right) \quad PE_{(pos,2i+1)}=\cos\!\left(\dfrac{pos}{10000^{2i/d}}\right)$",
}

if __name__ == "__main__":
    for name, latex in EQS.items():
        render(latex, f"{name}.png", fontsize=15)
        print("wrote", name)
