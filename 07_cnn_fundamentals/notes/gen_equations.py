import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from _shared.mathfig import render

EQS = {
    "eq_conv": r"$(I * K)[i,j] = \sum_m \sum_n I[i+m,\,j+n]\,K[m,n]$",
    "eq_outsize": r"$\mathrm{out} = \left\lfloor \dfrac{\mathrm{in} + 2p - k}{s} \right\rfloor + 1$",
    "eq_receptive": r"$R_l = R_{l-1} + (k_l - 1)\prod_{i<l} s_i$",
}

if __name__ == "__main__":
    for name, latex in EQS.items():
        render(latex, f"{name}.png", fontsize=17)
        print("wrote", name)
