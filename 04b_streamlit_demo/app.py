"""Bonus lesson: Streamlit, taught by making three of this course's own
results INTERACTIVE instead of static PDF figures.

Run it (after the repo-root Setup in the main README.md):
  cd 04b_streamlit_demo
  streamlit run app.py

Three tabs, each reusing code/data already built earlier in this course:
  1. Perceptron vs MLP (Days 3-4)  - live decision boundary, retrain on demand
  2. Optimizer race (Day 5)         - scrub through training steps
  3. Shape classifier + Grad-CAM (Day 15) - generate a shape, see the model look at it
"""
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from _shared.synthetic_data import make_moons, make_linearly_separable, make_xor, make_shape_images

st.set_page_config(page_title="Deep Learning Course — Interactive Demos", layout="wide")
st.title("Deep Learning Course — Interactive Demos")
st.caption(
    "A Streamlit bonus lesson: the same models from Days 3, 4, 5, and 15, "
    "made interactive instead of static PDF figures."
)

tab1, tab2, tab3 = st.tabs([
    "1. Perceptron vs MLP (Day 3-4)",
    "2. Optimizer race (Day 5)",
    "3. Shape classifier + Grad-CAM (Day 15)",
])

# ---------------------------------------------------------------------
# Tab 1: Perceptron vs MLP live decision boundary
# ---------------------------------------------------------------------
with tab1:
    st.markdown(
        "Day 3's perceptron hit a hard ceiling on non-linearly-separable data; "
        "Day 4's MLP broke through it. Pick a dataset and model, then retrain live."
    )
    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        dataset_name = st.selectbox("Dataset", ["moons", "linearly_separable", "xor"], index=0)
        model_type = st.radio("Model", ["Perceptron (1 layer)", "MLP (1 hidden layer)"], index=1)
        hidden_width = st.slider("Hidden width (MLP only)", 2, 32, 8, disabled=(model_type.startswith("Perceptron")))
        lr = st.slider("Learning rate", 0.01, 2.0, 0.5, step=0.01)
        epochs = st.slider("Epochs", 10, 500, 150, step=10)
        train_clicked = st.button("Train", type="primary")

    if "t1_result" not in st.session_state:
        st.session_state.t1_result = None

    if train_clicked:
        torch.manual_seed(0)
        if dataset_name == "moons":
            X, y = make_moons(400, seed=0)
        elif dataset_name == "linearly_separable":
            X, y = make_linearly_separable(400, seed=0)
        else:
            X, y = make_xor(400, seed=0)
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)

        if model_type.startswith("Perceptron"):
            model = nn.Linear(2, 1)
        else:
            model = nn.Sequential(nn.Linear(2, hidden_width), nn.Tanh(), nn.Linear(hidden_width, 1))

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        losses = []
        for _ in range(epochs):
            logits = model(Xt).squeeze(-1)
            loss = loss_fn(logits, yt)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        with torch.no_grad():
            acc = ((torch.sigmoid(model(Xt).squeeze(-1)) > 0.5).float() == yt).float().mean().item()

        xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 150),
                              np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 150))
        grid = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)
        with torch.no_grad():
            zz = torch.sigmoid(model(grid)).reshape(xx.shape).numpy()

        st.session_state.t1_result = dict(X=X, y=y, xx=xx, yy=yy, zz=zz, acc=acc, losses=losses,
                                            model_type=model_type, dataset_name=dataset_name)

    with col_plot:
        r = st.session_state.t1_result
        if r is None:
            st.info("Choose settings on the left and click Train.")
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
            ax1.contourf(r["xx"], r["yy"], r["zz"], levels=20, cmap="coolwarm", alpha=0.6)
            ax1.scatter(r["X"][:, 0], r["X"][:, 1], c=r["y"], cmap="coolwarm", s=10, edgecolor="k", linewidth=0.2)
            ax1.set_title(f"{r['model_type']} on {r['dataset_name']} — acc {r['acc']:.3f}")
            ax2.plot(r["losses"], color="#2c3e50")
            ax2.set_title("training loss"); ax2.set_xlabel("epoch")
            fig.tight_layout()
            st.pyplot(fig)
            st.metric("Final accuracy", f"{r['acc']:.1%}")

# ---------------------------------------------------------------------
# Tab 2: Optimizer race, scrubbable
# ---------------------------------------------------------------------
with tab2:
    st.markdown(
        "Day 5's ill-conditioned loss surface (`0.05x² + 5y²`) — steep in one "
        "direction, shallow in the other. Scrub through training steps to see "
        "each optimizer's path unfold, not just the final snapshot."
    )

    @st.cache_data
    def run_optimizer_paths(lr_sgd, lr_mom, lr_adam, steps):
        def loss_fn(p):
            return 0.05 * p[0] ** 2 + 5.0 * p[1] ** 2

        def run(name, lr, momentum=0.0):
            p = torch.tensor([-4.0, 1.0], requires_grad=True)
            if name == "adam":
                opt = torch.optim.Adam([p], lr=lr)
            else:
                opt = torch.optim.SGD([p], lr=lr, momentum=momentum)
            path = [p.detach().numpy().copy()]
            for _ in range(steps):
                opt.zero_grad()
                loss_fn(p).backward()
                opt.step()
                path.append(p.detach().numpy().copy())
            return np.array(path)

        return {
            "sgd": run("sgd", lr_sgd),
            "sgd_momentum": run("sgd_momentum", lr_mom, momentum=0.9),
            "adam": run("adam", lr_adam),
        }

    col_ctrl2, col_plot2 = st.columns([1, 2])
    with col_ctrl2:
        lr_sgd = st.slider("SGD learning rate", 0.01, 0.5, 0.19, step=0.01)
        lr_mom = st.slider("SGD+momentum learning rate", 0.01, 0.5, 0.19, step=0.01)
        lr_adam = st.slider("Adam learning rate", 0.05, 1.0, 0.3, step=0.05)
        total_steps = 60
        step = st.slider("Show steps up to", 1, total_steps, total_steps)

    paths = run_optimizer_paths(lr_sgd, lr_mom, lr_adam, total_steps)

    with col_plot2:
        xx, yy = np.meshgrid(np.linspace(-5, 2, 200), np.linspace(-2, 2, 200))
        zz = 0.05 * xx ** 2 + 5.0 * yy ** 2
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.contour(xx, yy, zz, levels=25, cmap="Greys", linewidths=0.6)
        colors = {"sgd": "#c0392b", "sgd_momentum": "#e67e22", "adam": "#27ae60"}
        for name, path in paths.items():
            p = path[:step + 1]
            ax.plot(p[:, 0], p[:, 1], "-o", markersize=2.5, linewidth=1, color=colors[name], label=name)
        ax.plot(0, 0, "k*", markersize=14, label="minimum")
        ax.set_xlabel("x (shallow)"); ax.set_ylabel("y (steep)")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

# ---------------------------------------------------------------------
# Tab 3: Shape classifier + Grad-CAM
# ---------------------------------------------------------------------
with tab3:
    st.markdown(
        "A lighter version of Day 15's capstone CNN (fewer epochs, smaller "
        "net, for a responsive demo — same shape-classification task and the "
        "same Grad-CAM interpretability check). Generate a new random shape "
        "and see where the model's decision comes from."
    )

    class DemoCNN(nn.Module):
        def __init__(self, n_classes=3):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.classifier = nn.Linear(32 * 8 * 8, n_classes)

        def forward(self, x, return_features=False):
            feats = self.features(x)
            logits = self.classifier(feats.flatten(1))
            return (logits, feats) if return_features else logits

    @st.cache_resource
    def train_demo_model():
        torch.manual_seed(0)
        X, y = make_shape_images(1200, size=32, seed=0)
        y = y.long()
        model = DemoCNN()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(8):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 64):
                idx = perm[i:i + 64]
                loss = loss_fn(model(X[idx]), y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        return model

    def grad_cam(model, x_single, target_class):
        x_single = x_single.unsqueeze(0)
        logits, feats = model(x_single, return_features=True)
        feats.retain_grad()
        logits[0, target_class].backward()
        weights = feats.grad[0].mean(dim=(1, 2))
        cam = F.relu((weights[:, None, None] * feats[0]).sum(0))
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam[None, None], size=32, mode="bilinear", align_corners=False)[0, 0]
        return cam.detach().numpy()

    model = train_demo_model()
    names = ["circle", "square", "triangle"]

    if "t3_seed" not in st.session_state:
        st.session_state.t3_seed = 0

    col_ctrl3, col_plot3 = st.columns([1, 2])
    with col_ctrl3:
        if st.button("Generate new random shape", type="primary"):
            st.session_state.t3_seed += 1

    img, label = make_shape_images(1, size=32, seed=1000 + st.session_state.t3_seed)
    with torch.no_grad():
        pred = model(img).argmax(-1).item()
    cam = grad_cam(model, img[0], target_class=pred)

    with col_plot3:
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(6, 3.2))
        axA.imshow(img[0, 0].numpy(), cmap="gray")
        axA.set_title(f"true={names[label.item()]}  pred={names[pred]}", fontsize=10)
        axA.axis("off")
        axB.imshow(img[0, 0].numpy(), cmap="gray")
        axB.imshow(cam, cmap="jet", alpha=0.5)
        axB.set_title("Grad-CAM", fontsize=10)
        axB.axis("off")
        fig.tight_layout()
        st.pyplot(fig)
        correct = (pred == label.item())
        st.metric("Prediction", names[pred], delta="correct" if correct else "WRONG",
                   delta_color="normal" if correct else "inverse")
