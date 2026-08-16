"""Render a LaTeX-style math expression to a standalone PNG using
matplotlib's built-in mathtext engine.

Why this exists: this course's PDFs were built with a markdown -> weasyprint
pipeline with no pandoc/LaTeX engine, and weasyprint cannot render $...$ math
natively. matplotlib's mathtext supports a large, genuinely useful subset
of LaTeX math syntax (fractions, subscripts/superscripts,
Greek letters, sums, partial derivatives, \\mathbf, \\nabla, etc.) without
needing a real TeX installation, so every equation in this course is
rendered once to a small PNG and embedded like any other figure.

Usage (from a day folder):
    from _shared.mathfig import render
    render(r"$\\frac{\\partial L}{\\partial w} = \\sum_i (\\hat{y}_i - y_i)\\, x_i$",
           "notes/eq_grad_mse.png")
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(latex: str, out_path: str, fontsize: int = 20, dpi: int = 200):
    """Render one math expression (wrap in $...$) to a tightly-cropped PNG."""
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, latex, fontsize=fontsize)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08,
                transparent=False, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    render(r"$\hat{y} = \sigma(w^\top x + b)$", "/tmp/test_eq.png")
    print("wrote /tmp/test_eq.png")
