"""Every dataset used in this 15-day course is generated in memory, on the
fly, from a fixed seed. No external downloads, no dataset files committed
to disk (this machine runs close to its disk quota, see ../README.md).
This also has a real pedagogical benefit used throughout the notes: when
you generate the data yourself, you know the true underlying function/
distribution, so you can directly check whether the network learned the
right thing instead of only trusting a held-out score.

Every generator below returns plain numpy arrays (X, y) or torch tensors,
documented per function. Import as:  from _shared.synthetic_data import ...
"""
import numpy as np
import torch


def make_moons(n=1000, noise=0.15, seed=0):
    """Two interleaving half-moons - the classic *linearly INseparable*
    2D classification toy set (used in Day 3/4 to motivate hidden layers).
    Hand-rolled (no sklearn) - see the notes for the parametric derivation."""
    rng = np.random.default_rng(seed)
    n0, n1 = n // 2, n - n // 2
    theta0 = rng.uniform(0, np.pi, n0)
    x0 = np.stack([np.cos(theta0), np.sin(theta0)], axis=1)
    theta1 = rng.uniform(0, np.pi, n1)
    x1 = np.stack([1 - np.cos(theta1), 1 - np.sin(theta1) - 0.5], axis=1)
    X = np.concatenate([x0, x1], axis=0)
    X += rng.normal(scale=noise, size=X.shape)
    y = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(np.float32)
    return X.astype(np.float32), y


def make_linearly_separable(n=1000, seed=0):
    """Two Gaussian blobs, far enough apart to be linearly separable -
    the case a single perceptron (Day 3) CAN solve, contrasted with
    make_moons/make_xor which it cannot."""
    rng = np.random.default_rng(seed)
    n0, n1 = n // 2, n - n // 2
    x0 = rng.normal(loc=[-2, -2], scale=0.9, size=(n0, 2))
    x1 = rng.normal(loc=[2, 2], scale=0.9, size=(n1, 2))
    X = np.concatenate([x0, x1], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(np.float32)
    return X, y


def make_xor(n=1000, noise=0.15, seed=0):
    """4-cluster XOR pattern - not linearly separable by ANY single line,
    the canonical proof-by-example that motivates multi-layer networks
    (Day 4)."""
    rng = np.random.default_rng(seed)
    centers = np.array([[-2, -2], [-2, 2], [2, -2], [2, 2]], dtype=np.float32)
    labels = np.array([0, 1, 1, 0], dtype=np.float32)  # XOR of quadrant signs
    idx = rng.integers(0, 4, size=n)
    X = centers[idx] + rng.normal(scale=noise * 3, size=(n, 2)).astype(np.float32)
    y = labels[idx]
    return X, y


def make_polynomial_regression(n=60, degree_true=3, noise=0.3, seed=0):
    """1D regression target y = 0.5x^3 - x^2 - x + noise, sampled on a
    small n (Day 6) so overfitting a high-capacity model is easy to
    demonstrate and easy to see on a single held-out curve."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.5, 2.5, size=n).astype(np.float32)
    y_true = 0.5 * x**3 - x**2 - x
    y = (y_true + rng.normal(scale=noise, size=n)).astype(np.float32)
    return x.reshape(-1, 1), y


def make_shape_images(n=2000, size=32, seed=0):
    """Tiny synthetic 3-class image dataset: filled circle / square /
    triangle, random position/size/rotation/gray-level jitter, rendered
    with PIL. Stands in for MNIST/CIFAR (Day 7/8/12/13/14/15) without
    downloading anything - a few thousand 32x32x1 images is a handful of
    MB in memory and is never written to disk as a full dataset (only a
    few example PNGs are saved for the notes figures).
    Returns: images float32 tensor [n,1,size,size] in [-1,1], labels long
    tensor [n] in {0:circle, 1:square, 2:triangle}."""
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, size, size), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        cls = rng.integers(0, 3)
        labels[i] = cls
        img = Image.new("L", (size, size), color=int(rng.integers(210, 256)))
        draw = ImageDraw.Draw(img)
        cx, cy = rng.uniform(size * 0.35, size * 0.65, 2)
        r = rng.uniform(size * 0.18, size * 0.32)
        ink = int(rng.integers(0, 60))
        if cls == 0:  # circle
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ink)
        elif cls == 1:  # square
            ang = rng.uniform(0, 45)
            box = Image.new("L", (size, size), 255)
            bd = ImageDraw.Draw(box)
            bd.rectangle([cx - r, cy - r, cx + r, cy + r], fill=ink)
            box = box.rotate(ang, fillcolor=255)
            img = Image.composite(box, img, box.point(lambda p: 255 - p))
        else:  # triangle
            ang0 = rng.uniform(0, 360)
            pts = [(cx + r * np.cos(np.radians(ang0 + k * 120)),
                    cy + r * np.sin(np.radians(ang0 + k * 120))) for k in range(3)]
            draw.polygon(pts, fill=ink)
        arr = np.asarray(img, dtype=np.float32)
        noise = rng.normal(scale=4.0, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255)
        imgs[i] = arr
    X = torch.from_numpy(imgs).unsqueeze(1) / 127.5 - 1.0  # [-1,1]
    y = torch.from_numpy(labels)
    return X, y


def make_sine_sequence(n_sequences=500, seq_len=20, seed=0):
    """Next-step sine-wave prediction task (Day 9, RNN/LSTM): each sequence
    is seq_len consecutive samples of a sine wave with a random phase and
    frequency; the target at each step is the NEXT sample (teacher-forced
    sequence regression). Returns X [n,seq_len,1], y [n,seq_len,1]."""
    rng = np.random.default_rng(seed)
    t = np.arange(seq_len + 1)
    X = np.zeros((n_sequences, seq_len, 1), dtype=np.float32)
    Y = np.zeros((n_sequences, seq_len, 1), dtype=np.float32)
    for i in range(n_sequences):
        freq = rng.uniform(0.15, 0.4)
        phase = rng.uniform(0, 2 * np.pi)
        wave = np.sin(freq * t + phase)
        X[i, :, 0] = wave[:-1]
        Y[i, :, 0] = wave[1:]
    return torch.from_numpy(X), torch.from_numpy(Y)


def make_copy_task(n=2000, seq_len=8, vocab=10, seed=0):
    """Sequence-to-sequence COPY task (Day 10, attention/Transformers):
    input is a random token sequence, target is the identical sequence.
    Trivial for attention (direct token-to-token alignment) and a clean
    way to see attention weights line up on the diagonal. Tokens are
    ints in [1, vocab-1]; 0 is reserved as a padding/start token.
    Returns: src, tgt as long tensors [n, seq_len]."""
    rng = np.random.default_rng(seed)
    src = rng.integers(1, vocab, size=(n, seq_len)).astype(np.int64)
    tgt = src.copy()
    return torch.from_numpy(src), torch.from_numpy(tgt)


def make_char_corpus(seed=0):
    """A tiny fixed character-level text corpus (Day 9 alt task / general
    tokenization demos) - short enough to print in notes, long enough to
    show real char-bigram structure. A couple of hand-written sentences
    about deep learning itself, rather than pulling in an external text
    file."""
    return (
        "deep learning turns raw data into useful representations. "
        "a neural network is just a differentiable function with "
        "parameters we adjust using gradients. gradients tell us how "
        "to change each weight to reduce the loss. training is search "
        "guided by calculus, not magic."
    )


def to_loader(X, y, batch_size=32, shuffle=True):
    """Small convenience wrapper so every day's code can go straight from
    a synthetic_data.make_*() call to a DataLoader in one line."""
    ds = torch.utils.data.TensorDataset(
        X if torch.is_tensor(X) else torch.from_numpy(X),
        y if torch.is_tensor(y) else torch.from_numpy(y))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
