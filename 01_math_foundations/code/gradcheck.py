"""Day 1 hands-on: prove the chain rule to yourself with numbers, not faith.

We define a tiny composite function by hand (the same shape as one neuron's
forward pass: linear -> sigmoid -> squared-error loss against a target),
compute its gradient two completely independent ways, and check they agree
to numerical precision:

  1. ANALYTIC gradient - derived on paper using the chain rule, coded
     directly as a formula (this is exactly what backprop automates).
  2. NUMERICAL gradient - the definition of a derivative itself:
     f'(x) ~= (f(x+h) - f(x-h)) / (2h) for a tiny h (central difference).

If these two disagree by more than ~1e-4, either the calculus is wrong or
the code implementing it is wrong. This exact technique (gradcheck) is a
real, standard debugging tool used when implementing any new layer by
hand - PyTorch's own test suite uses torch.autograd.gradcheck for this.
"""
import numpy as np

rng = np.random.default_rng(0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def forward(w, b, x, y_target):
    """One 'neuron': z = w.x + b, y_hat = sigmoid(z), L = 0.5*(y_hat - y_target)^2"""
    z = np.dot(w, x) + b
    y_hat = sigmoid(z)
    L = 0.5 * (y_hat - y_target) ** 2
    return L, y_hat, z


def analytic_grad_w(w, b, x, y_target):
    """dL/dw via the chain rule:
       dL/dy_hat   = (y_hat - y_target)
       dy_hat/dz   = y_hat * (1 - y_hat)          [sigmoid derivative]
       dz/dw       = x
       dL/dw       = dL/dy_hat * dy_hat/dz * dz/dw
    """
    L, y_hat, z = forward(w, b, x, y_target)
    dL_dyhat = (y_hat - y_target)
    dyhat_dz = y_hat * (1 - y_hat)
    dz_dw = x
    return dL_dyhat * dyhat_dz * dz_dw  # chain rule, three factors multiplied


def numerical_grad_w(w, b, x, y_target, h=1e-5):
    """Central-difference estimate of dL/dw_i for every component i,
    perturbing one weight at a time and holding everything else fixed."""
    grad = np.zeros_like(w)
    for i in range(len(w)):
        w_plus = w.copy(); w_plus[i] += h
        w_minus = w.copy(); w_minus[i] -= h
        L_plus, _, _ = forward(w_plus, b, x, y_target)
        L_minus, _, _ = forward(w_minus, b, x, y_target)
        grad[i] = (L_plus - L_minus) / (2 * h)
    return grad


def main():
    x = rng.normal(size=5)
    w = rng.normal(size=5)
    b = rng.normal()
    y_target = 1.0

    g_analytic = analytic_grad_w(w, b, x, y_target)
    g_numeric = numerical_grad_w(w, b, x, y_target)

    print("analytic gradient: ", np.round(g_analytic, 6))
    print("numerical gradient:", np.round(g_numeric, 6))
    max_diff = np.max(np.abs(g_analytic - g_numeric))
    print(f"max |analytic - numeric| = {max_diff:.2e}")
    assert max_diff < 1e-6, "gradcheck FAILED - the calculus or the code has a bug"
    print("gradcheck PASSED: the chain-rule derivation matches the definition of a derivative.")


if __name__ == "__main__":
    main()
