"""Day 10 hands-on: implement scaled dot-product attention by hand (no
nn.MultiheadAttention), train a tiny transformer encoder on the COPY task
(Day 9's synthetic_data.make_copy_task - output should equal input,
token-for-token), and visualize the attention weights - they should line
up on the diagonal, since "copy token i" means "attend to position i".
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import math
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _shared.synthetic_data import make_copy_task

torch.manual_seed(0)


def scaled_dot_product_attention(Q, K, V, return_weights=False):
    """Q,K,V: [batch, seq_len, d_k]. Implements eq_attention directly."""
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # [batch, seq_len, seq_len]
    weights = torch.softmax(scores, dim=-1)             # each ROW sums to 1
    out = weights @ V
    return (out, weights) if return_weights else out


class TinySelfAttentionLayer(nn.Module):
    """One self-attention block: project x into Q,K,V with learned linear
    maps, attend, then a small feedforward - the two sub-layers every real
    transformer encoder block is built from (real ones also add layernorm
    around each, omitted here to keep the from-scratch attention mechanism
    itself the focus).

    attn_residual=True is the STANDARD, real-transformer configuration
    (residual around both sub-layers, per Day 8). attn_residual=False is a
    deliberate teaching variant used once below purely to visualize what
    attention alone learns - see notes.md for why this changes the
    attention-weight pattern so dramatically."""
    def __init__(self, d_model, d_ff=64, attn_residual=True):
        super().__init__()
        self.attn_residual = attn_residual
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, x, return_weights=False):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        attn_out, weights = scaled_dot_product_attention(Q, K, V, return_weights=True)
        x = (x + attn_out) if self.attn_residual else attn_out
        x = x + self.ff(x)        # residual around the feedforward, always
        return (x, weights) if return_weights else x


class TinyTransformerCopy(nn.Module):
    def __init__(self, vocab, d_model=32, seq_len=8, n_layers=2, attn_residual=True):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [TinySelfAttentionLayer(d_model, attn_residual=attn_residual) for _ in range(n_layers)])
        self.out_proj = nn.Linear(d_model, vocab)

    def forward(self, x, return_weights=False):
        h = self.embed(x) + self.pos_embed
        all_weights = []
        for layer in self.layers:
            h, w = layer(h, return_weights=True)
            all_weights.append(w)
        logits = self.out_proj(h)
        return (logits, all_weights) if return_weights else logits


def train_and_plot(attn_residual, out_path, seed=0):
    vocab, seq_len = 10, 8
    torch.manual_seed(seed)
    src, tgt = make_copy_task(n=2000, seq_len=seq_len, vocab=vocab)
    src_train, tgt_train = src[:1600], tgt[:1600]
    src_val, tgt_val = src[1600:], tgt[1600:]

    torch.manual_seed(seed)
    model = TinyTransformerCopy(vocab, seq_len=seq_len, attn_residual=attn_residual)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    tag = "WITH residual around attention (standard)" if attn_residual else "NO residual around attention (teaching variant)"
    print(f"=== {tag} ===")
    for epoch in range(60):
        model.train()
        perm = torch.randperm(len(src_train))
        for i in range(0, len(src_train), 64):
            idx = perm[i:i + 64]
            logits = model(src_train[idx])                      # [batch, seq_len, vocab]
            loss = loss_fn(logits.reshape(-1, vocab), tgt_train[idx].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 15 == 0 or epoch == 59:
            model.eval()
            with torch.no_grad():
                pred = model(src_val).argmax(-1)
                token_acc = (pred == tgt_val).float().mean().item()
                exact_acc = (pred == tgt_val).all(dim=1).float().mean().item()
            print(f"  epoch {epoch:2d}  token_acc {token_acc:.3f}  exact_sequence_acc {exact_acc:.3f}")

    model.eval()
    example = src_val[0:1]
    with torch.no_grad():
        _, all_weights = model(example, return_weights=True)

    fig, axes = plt.subplots(1, len(all_weights), figsize=(4.5 * len(all_weights), 4))
    if len(all_weights) == 1:
        axes = [axes]
    for i, (ax, w) in enumerate(zip(axes, all_weights)):
        im = ax.imshow(w[0].detach().numpy(), cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"layer {i+1} attention weights")
        ax.set_xlabel("attending TO position (key)")
        ax.set_ylabel("attending FROM position (query)")
    fig.suptitle(tag, fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8, label="attention weight")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"wrote {out_path}\n")


if __name__ == "__main__":
    train_and_plot(attn_residual=True, out_path="../notes/attention_weights_with_residual.png")
    train_and_plot(attn_residual=False, out_path="../notes/attention_weights_no_residual.png")
