# Deep Learning with PyTorch — a 15-Day Course

![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)
![Days](https://img.shields.io/badge/days-15%20%2B%20bonus-brightgreen)
![Streamlit](https://img.shields.io/badge/bonus%20lesson-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Last commit](https://img.shields.io/github/last-commit/meharuhanzz/deep-learning-15days)
![Repo size](https://img.shields.io/github/repo-size/meharuhanzz/deep-learning-15days)
![License](https://img.shields.io/github/license/meharuhanzz/deep-learning-15days)

A from-first-principles deep learning course: mathematical derivations,
runnable PyTorch code, and honestly-reported experiment results for every
topic — verify, don't assert; report what actually happened, including
when it doesn't match the textbook story on the first attempt.

## How this course is organized

Every day is a self-contained folder:

```
NN_topic_name/
  notes.md          the day's material: theory, math, code walkthroughs, insights
  notes.pdf          the same content, pre-rendered
  notes/             rendered equation images + generated figures, embedded in notes.md
  code/              runnable Python scripts that produced every number/figure in notes.md
```

`_shared/` holds two modules every day imports:

- **`mathfig.py`** — renders LaTeX-style math to PNG via matplotlib's
  mathtext engine (no LaTeX/pandoc installation needed).
- **`synthetic_data.py`** — every dataset used across all 15 days,
  generated in memory from a fixed seed. **No datasets are downloaded or
  stored on disk** (aside from two small pretrained model checkpoints in
  Days 8 and 11, a few tens of MB total) — this machine runs close to its
  disk quota, so every dataset here is either hand-generated (2D toy
  distributions, synthetic shape images, synthetic sequences) or a
  deliberately tiny real download. This has a real pedagogical upside
  used throughout the notes: generating your own data means you know the
  *true* underlying rule, so any gap between train and real performance
  is unambiguous.

## Setup

```
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install torch torchvision numpy matplotlib pillow markdown weasyprint \
            transformers sentencepiece streamlit
```

Any single day only needs a subset of these (e.g. Day 1 needs nothing but
`numpy`/`matplotlib`); the full list above covers everything through
Day 15 and the Streamlit bonus lesson. GPU is optional — every script
runs on CPU, just slower for the CNN/transformer days.

## The 15 days

| Day | Topic | Core new idea |
|---|---|---|
| 1 | Math Foundations | Vectors/matrices, gradients, the chain rule, gradient-checking |
| 2 | PyTorch Fundamentals | Tensors, autograd, the define-by-run computation graph |
| 3 | Perceptron & Gradient Descent | The training-loop pattern; linear models' hard ceiling |
| 4 | MLP & Backpropagation | Backprop derived and verified by hand; vanishing gradients, part 1 (activation) |
| *bonus* | *Streamlit* | *`04b_streamlit_demo/` — Days 3/4/5/15's models made interactive (not one of the 15 numbered days)* |
| 5 | Losses & Optimizers | Cross-entropy as likelihood; momentum and Adam vs. plain SGD |
| 6 | Regularization & Generalization | Overfitting made visible; vanishing gradients, part 2 (initialization) |
| 7 | CNN Fundamentals | Convolution as parameter sharing; CNN vs. MLP, measured |
| 8 | CNN Architectures & Transfer Learning | Residual connections (vanishing gradients, part 3); reusing pretrained features |
| 9 | RNNs & LSTMs | Vanishing gradients across *time*; LSTM's gated additive shortcut |
| 10 | Attention & Transformers | Scaled dot-product attention from scratch; O(1) gradient paths |
| 11 | Hugging Face Ecosystem | Subword tokenization; fine-tuning a real pretrained transformer |
| 12 | Autoencoders & VAEs | Bottleneck compression; the reparameterization trick; sampling |
| 13 | GANs & Diffusion | Adversarial training vs. iterative denoising, compared head-to-head |
| 14 | Training at Scale | Mixed precision, gradient accumulation, LR schedules — measured on this GPU |
| 15 | Capstone | Every technique combined into one pipeline; honest held-out evaluation; Grad-CAM |

**A thread running through Days 4, 6, 8, and 9**: the vanishing-gradient
problem is introduced, measured, and then fixed by three genuinely
different, complementary mechanisms (activation choice, weight
initialization, residual/gated shortcuts) — each demonstrated with an
actual before/after gradient measurement, not asserted. Day 9 shows the
identical phenomenon recurring across time instead of depth. This isn't
incidental repetition; it's the course's central worked example of how
one underlying mechanism (repeated multiplication of small/large factors
through backpropagation) explains failure modes that look superficially
unrelated across architectures.

Every day's code was developed and verified on an RTX 5090 (PyTorch
2.11+cu128), but nothing in the course requires a GPU specifically — see
Setup above.
