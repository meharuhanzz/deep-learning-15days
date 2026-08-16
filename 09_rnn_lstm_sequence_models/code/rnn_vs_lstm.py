"""Day 9 hands-on:
  (a) measure how a gradient shrinks as it's backpropagated THROUGH TIME
      in a plain RNN vs an LSTM, on a long sequence - the exact same kind
      of measurement Days 4/6/8 made across DEPTH, now made across time.
  (b) train both on a long-range copy task (remember a signal seen many
      steps ago) to see the practical consequence.
"""
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0)


def gradient_through_time(cell_type="rnn", seq_len=50, hidden=32):
    """Feed a fixed random sequence through an RNN/LSTM cell manually
    unrolled seq_len times, backprop from the FINAL hidden state, and
    record the gradient norm reaching the hidden state at each earlier
    timestep - directly analogous to Days 4/6/8's per-layer measurement."""
    torch.manual_seed(1)
    x_seq = torch.randn(seq_len, 1, 4)  # [time, batch, features]

    if cell_type == "rnn":
        cell = nn.RNNCell(4, hidden)
        h = torch.zeros(1, hidden, requires_grad=True)
        states = [h]
        for t in range(seq_len):
            h = cell(x_seq[t], h)
            h.retain_grad()
            states.append(h)
        loss = h.sum()
        loss.backward()
        return [s.grad.norm().item() for s in states if s.grad is not None]

    elif cell_type == "lstm":
        cell = nn.LSTMCell(4, hidden)
        h = torch.zeros(1, hidden, requires_grad=True)
        c = torch.zeros(1, hidden, requires_grad=True)
        states = [h]
        for t in range(seq_len):
            h, c = cell(x_seq[t], (h, c))
            h.retain_grad()
            states.append(h)
        loss = h.sum()
        loss.backward()
        return [s.grad.norm().item() for s in states if s.grad is not None]


def plot_gradient_decay():
    rnn_grads = gradient_through_time("rnn")
    lstm_grads = gradient_through_time("lstm")
    print(f"RNN : grad norm at t=0 (50 steps back) = {rnn_grads[0]:.2e}")
    print(f"LSTM: grad norm at t=0 (50 steps back) = {lstm_grads[0]:.2e}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(range(len(rnn_grads)), rnn_grads, color="#c0392b", label="plain RNN")
    ax.plot(range(len(lstm_grads)), lstm_grads, color="#27ae60", label="LSTM")
    ax.set_yscale("log")
    ax.set_xlabel("timestep (0 = furthest in the past, 50 = present)")
    ax.set_ylabel("gradient norm reaching this timestep's hidden state")
    ax.set_title("Backprop through TIME: same phenomenon as through depth", fontsize=11)
    ax.legend()
    fig.tight_layout()
    fig.savefig("../notes/gradient_through_time.png", dpi=130)
    print("wrote ../notes/gradient_through_time.png")


def long_range_copy_task(n=800, seq_len=150, seed=0):
    """The signal to predict appears ONLY at position 0; everything else
    is noise. A model must carry that one value forward through the
    entire sequence to predict it correctly at the end - a direct test of
    long-range memory."""
    rng = torch.Generator().manual_seed(seed)
    signal = torch.randint(0, 2, (n, 1), generator=rng).float()  # the value to remember
    noise = torch.randn(n, seq_len - 1, 1, generator=rng) * 0.5
    seq = torch.cat([signal.unsqueeze(1), noise], dim=1)  # [n, seq_len, 1]: signal at t=0, noise after
    return seq, signal.squeeze(-1)


class RNNClassifier(nn.Module):
    def __init__(self, cell_type, hidden=32):
        super().__init__()
        self.rnn = nn.RNN(1, hidden, batch_first=True) if cell_type == "rnn" \
            else nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        if cell_type == "lstm":
            # Standard, well-documented LSTM init trick (Jozefowicz et al.
            # 2015): initialize the forget gate's bias to 1, so the cell
            # starts training in "remember by default" mode instead of
            # PyTorch's default zero-init, which starts every gate at 0.5
            # open and can make early training unnecessarily unstable on
            # long sequences. PyTorch packs LSTM biases as 4 chunks in
            # [input, forget, cell, output] gate order.
            for name, param in self.rnn.named_parameters():
                if "bias" in name:
                    n = param.size(0) // 4
                    param.data[n:2 * n].fill_(1.0)

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]  # final timestep's hidden state must contain the memory
        return self.head(last).squeeze(-1)


def train_copy_task():
    X, y = long_range_copy_task(n=800, seq_len=150)
    X_train, y_train = X[:640], y[:640]
    X_val, y_val = X[640:], y[640:]

    results = {}
    for cell_type in ["rnn", "lstm"]:
        torch.manual_seed(0)
        model = RNNClassifier(cell_type)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        accs = []
        for epoch in range(40):
            model.train()
            for i in range(0, 640, 32):
                xb, yb = X_train[i:i + 32], y_train[i:i + 32]
                logits = model(xb)
                loss = loss_fn(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                acc = ((torch.sigmoid(model(X_val)) > 0.5).float() == y_val).float().mean().item()
            accs.append(acc)
        results[cell_type] = accs
        print(f"{cell_type}: final val acc on 150-step long-range copy task = {accs[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(results["rnn"], color="#c0392b", label="plain RNN")
    ax.plot(results["lstm"], color="#27ae60", label="LSTM")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance level")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation accuracy")
    ax.set_title("Long-range copy task (remember signal from 150 steps back)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("../notes/long_range_copy.png", dpi=130)
    print("wrote ../notes/long_range_copy.png")


if __name__ == "__main__":
    plot_gradient_decay()
    print()
    train_copy_task()
