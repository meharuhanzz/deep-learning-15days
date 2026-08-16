"""Day 4 hands-on, part 1: backpropagation through a 2-layer MLP,
implemented BY HAND with raw NumPy arrays (no autograd) - then verified
against PyTorch autograd, continuing the Day 1/2 "prove it, don't trust
it" theme one level deeper: this time the object under test is a network
with a HIDDEN layer, not a single neuron.

Architecture:  x -> Linear(W1,b1) -> tanh -> Linear(W2,b2) -> sigmoid -> L

Forward:
  z1 = W1 x + b1 ;  h = tanh(z1)
  z2 = W2 h + b2 ;  y_hat = sigmoid(z2)
  L  = 0.5 (y_hat - y)^2

Backward (see notes.md for the full derivation):
  delta2 = (y_hat - y) * y_hat * (1 - y_hat)          # sigmoid + MSE
  dL/dW2 = delta2 @ h.T ;  dL/db2 = delta2
  delta1 = (W2.T @ delta2) * (1 - h**2)               # tanh derivative
  dL/dW1 = delta1 @ x.T ;  dL/db1 = delta1
"""
import numpy as np
import torch

rng = np.random.default_rng(0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def forward(params, x):
    W1, b1, W2, b2 = params
    z1 = W1 @ x + b1
    h = np.tanh(z1)
    z2 = W2 @ h + b2
    y_hat = sigmoid(z2)
    cache = (x, z1, h, z2, y_hat)
    return y_hat, cache


def backward(params, cache, y_target):
    """Every line here is one factor of the chain rule from Day 1,
    applied twice (once per layer) - this IS backpropagation, with no
    framework hiding the mechanics."""
    W1, b1, W2, b2 = params
    x, z1, h, z2, y_hat = cache

    # --- output layer -------------------------------------------------
    dL_dyhat = (y_hat - y_target)                  # dL/dy_hat, L = 0.5(y_hat-y)^2
    dyhat_dz2 = y_hat * (1 - y_hat)                 # sigmoid'(z2)
    delta2 = dL_dyhat * dyhat_dz2                   # dL/dz2  (chain rule)
    dW2 = np.outer(delta2, h)                       # dL/dW2 = delta2 . h^T
    db2 = delta2                                    # dL/db2 = delta2

    # --- hidden layer: propagate the error BACKWARD through W2 --------
    dh = W2.T @ delta2                              # dL/dh, via W2's transpose
    dz1 = dh * (1 - h ** 2)                          # tanh'(z1) = 1 - tanh(z1)^2
    delta1 = dz1                                     # dL/dz1
    dW1 = np.outer(delta1, x)                        # dL/dW1 = delta1 . x^T
    db1 = delta1                                     # dL/db1 = delta1

    return [dW1, db1, dW2, db2]


def init_params(n_in=2, n_hidden=4, n_out=1):
    W1 = rng.normal(scale=0.5, size=(n_hidden, n_in))
    b1 = np.zeros(n_hidden)
    W2 = rng.normal(scale=0.5, size=(n_out, n_hidden))
    b2 = np.zeros(n_out)
    return [W1, b1, W2, b2]


def loss_fn(y_hat, y_target):
    return 0.5 * np.sum((y_hat - y_target) ** 2)


# ---------------------------------------------------------------------
# Verification: does the hand-derived backward() match torch.autograd,
# for the SAME weights and the SAME input? (Day 1/2's gradcheck pattern,
# now applied to a real 2-layer network instead of a single neuron.)
# ---------------------------------------------------------------------
def verify_against_autograd():
    params = init_params()
    x = rng.normal(size=2)
    y_target = np.array([1.0])

    y_hat, cache = forward(params, x)
    grads_manual = backward(params, cache, y_target)

    W1, b1, W2, b2 = [torch.tensor(p, requires_grad=True) for p in params]
    xt = torch.tensor(x)
    yt = torch.tensor(y_target)
    z1 = W1 @ xt + b1
    h = torch.tanh(z1)
    z2 = W2 @ h + b2
    y_hat_t = torch.sigmoid(z2)
    L = 0.5 * torch.sum((y_hat_t - yt) ** 2)
    L.backward()

    grads_auto = [W1.grad.numpy(), b1.grad.numpy(), W2.grad.numpy(), b2.grad.numpy()]
    names = ["dW1", "db1", "dW2", "db2"]
    print("=== hand-derived backprop vs torch.autograd, same weights/input ===")
    max_diff = 0.0
    for name, gm, ga in zip(names, grads_manual, grads_auto):
        d = np.max(np.abs(gm - ga))
        max_diff = max(max_diff, d)
        print(f"  {name}: max |manual - autograd| = {d:.2e}")
    assert max_diff < 1e-8, "hand-derived backprop disagrees with autograd - bug!"
    print(f"MATCH (worst case {max_diff:.2e}): the by-hand derivation is exactly backprop.\n")


# ---------------------------------------------------------------------
# Train the hand-written MLP on the two-moons data Day 3's single-layer
# perceptron could not solve (capped at ~87% accuracy there).
# ---------------------------------------------------------------------
def train_on_moons():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from _shared.synthetic_data import make_moons

    X, y = make_moons(400, seed=0)
    params = init_params(n_in=2, n_hidden=8, n_out=1)
    lr = 0.5

    print("=== training the hand-written 2-layer MLP on two-moons ===")
    for epoch in range(300):
        total_loss = 0.0
        correct = 0
        perm = rng.permutation(len(X))
        for i in perm:
            xi, yi = X[i], np.array([y[i]])
            y_hat, cache = forward(params, xi)
            total_loss += loss_fn(y_hat, yi)
            correct += int((y_hat[0] > 0.5) == (yi[0] > 0.5))
            grads = backward(params, cache, yi)
            for p, g in zip(params, grads):
                p -= lr * g   # plain SGD update, one sample at a time
        if epoch % 30 == 0 or epoch == 299:
            print(f"  epoch {epoch:3d}  avg loss {total_loss/len(X):.4f}  acc {correct/len(X):.3f}")

    acc = correct / len(X)
    print(f"\nfinal accuracy: {acc:.3f} (perceptron, Day 3, was stuck at ~0.87)")
    assert acc > 0.95, "expected the MLP to clearly beat the linear perceptron on moons"
    print("PASSED: a single hidden layer solved a problem no linear model could.")


if __name__ == "__main__":
    verify_against_autograd()
    train_on_moons()
