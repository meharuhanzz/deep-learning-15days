"""Day 2 hands-on: the exact same one-neuron function from Day 1
(linear -> sigmoid -> squared error), but instead of hand-deriving the
chain rule, we let torch.autograd do it - then check it produces the
SAME numbers ../../01_math_foundations/code/gradcheck.py computed by hand.

This is the whole point of today: autograd is not a different, mysterious
process from what you did on paper - it is the identical chain rule,
applied automatically to a dynamically-built computation graph.
"""
import torch

torch.manual_seed(0)


def main():
    x = torch.randn(5)
    w = torch.randn(5, requires_grad=True)   # track gradients w.r.t. w
    b = torch.randn(1, requires_grad=True)   # and w.r.t. b
    y_target = torch.tensor(1.0)

    # ---- forward pass: PyTorch silently builds a computation graph here.
    # Every operation (dot, +, sigmoid, subtract, square) becomes a node
    # that remembers (a) its inputs and (b) how to compute its own local
    # derivative. This is called "define-by-run": the graph exists only
    # for as long as this forward pass's tensors are alive, and a totally
    # different graph could be built on the next call (e.g. an RNN
    # unrolled a different number of steps). Contrast with a "static
    # graph" framework, which compiles the graph once ahead of time.
    z = torch.dot(w, x) + b
    y_hat = torch.sigmoid(z)
    L = 0.5 * (y_hat - y_target) ** 2

    print("forward: z=%.4f y_hat=%.4f L=%.6f" % (z.item(), y_hat.item(), L.item()))

    # ---- backward pass: walk the graph in reverse, multiplying local
    # derivatives via the chain rule at each node, accumulating the result
    # into every leaf tensor's .grad. This one call replaces every line of
    # analytic_grad_w() from Day 1.
    L.backward()

    print("autograd dL/dw:", w.grad.numpy())
    print("autograd dL/db:", b.grad.numpy())

    # ---- cross-check against Day 1's hand-derived formula on the SAME
    # numbers, to make the equivalence concrete rather than assumed.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[2] / "01_math_foundations" / "code"))
    from gradcheck import analytic_grad_w  # Day 1's hand-derived formula

    g_manual = analytic_grad_w(w.detach().numpy(), b.item(), x.numpy(), y_target.item())
    print("Day 1 manual dL/dw:", g_manual)
    max_diff = (w.grad.numpy() - g_manual).__abs__().max()
    print(f"max |autograd - manual| = {max_diff:.2e}")
    assert max_diff < 1e-5, "autograd and the hand-derived chain rule disagree - investigate!"
    print("MATCH: autograd reproduced exactly what the chain rule predicts by hand.")

    # ---- the accumulation gotcha: calling .backward() again WITHOUT
    # zeroing gradients first adds to the existing .grad, because PyTorch
    # doesn't know whether you want a fresh gradient or to accumulate one
    # across multiple backward calls (useful for gradient accumulation
    # over "micro-batches", Day 14). This is the single most common
    # PyTorch training bug for beginners.
    grad_before = w.grad.clone()
    L2 = 0.5 * (torch.sigmoid(torch.dot(w, x) + b) - y_target) ** 2
    L2.backward()
    print("\nafter a SECOND .backward() without zero_grad():")
    print("  w.grad is now", w.grad.numpy(), "= 2x the single-call gradient:", (grad_before * 2).numpy())
    print("  This is why every training loop calls optimizer.zero_grad() before .backward().")


if __name__ == "__main__":
    main()
