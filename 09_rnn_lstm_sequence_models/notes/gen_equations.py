import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from _shared.mathfig import render

EQS = {
    "eq_rnn_recurrence": r"$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$",
    "eq_bptt_product": r"$\dfrac{\partial L}{\partial h_0} = \dfrac{\partial L}{\partial h_T}\prod_{t=1}^{T} \dfrac{\partial h_t}{\partial h_{t-1}}$",
    "eq_lstm_gates": r"$f_t=\sigma(W_f[h_{t-1},x_t]),\ \ i_t=\sigma(W_i[h_{t-1},x_t]),\ \ o_t=\sigma(W_o[h_{t-1},x_t])$",
    "eq_lstm_cell": r"$c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t \qquad h_t = o_t \odot \tanh(c_t)$",
}

if __name__ == "__main__":
    for name, latex in EQS.items():
        render(latex, f"{name}.png", fontsize=15)
        print("wrote", name)
