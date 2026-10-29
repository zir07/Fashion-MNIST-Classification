"""Figures. Report figures are sized for the IEEE two-column layout (single column 3.45 in, double 7.1 in)."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from . import config as C
from .models import DIV, RAW, STD

SHOW = False            # notebooks set this to True to display figures inline

REPORT = {"font.size": 7, "font.family": "serif"}
SHORT = {RAW: "raw", STD: "standardised", DIV: "\u00f7255"}
COLORS = {RAW: "C1", STD: "C2", DIV: "C0"}


def _finish(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi)
    if SHOW:
        plt.show()
    else:
        plt.close(fig)


# ----------------------------------------------------------------------------- exploratory figures
def class_distribution(y, path):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(C.CLASS_NAMES, np.bincount(y, minlength=10), color="steelblue")
    ax.set_ylabel("Training images"); ax.set_title("Class distribution (training set)")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    fig.tight_layout()
    _finish(fig, path, dpi=200)


def sample_grid(X, y, path, n_per_class=8, seed=C.RANDOM_STATE):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(10, n_per_class, figsize=(n_per_class + 1.8, 10.0))
    for c in range(10):
        idx = rng.choice(np.flatnonzero(y == c), n_per_class, replace=False)
        for j, i in enumerate(idx):
            ax = axes[c, j]
            ax.imshow(X[i].reshape(28, 28), cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(C.CLASS_NAMES[c], rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle("Random training samples per class", y=0.995)
    fig.tight_layout()
    _finish(fig, path, dpi=200)


def mean_images(X, y, path):
    """Report Fig. 1: mean image per class (single column)."""
    with plt.rc_context(REPORT):
        fig, axes = plt.subplots(2, 5, figsize=(3.45, 1.55))
        for c, ax in enumerate(axes.ravel()):
            ax.imshow(X[y == c].mean(axis=0).reshape(28, 28), cmap="gray")
            ax.set_title(C.CLASS_NAMES[c], fontsize=6, pad=2)
            ax.axis("off")
        fig.subplots_adjust(wspace=0.05, hspace=0.25, left=0.01, right=0.99, top=0.9, bottom=0.01)
        _finish(fig, path)


def pixel_histogram(X, path):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(np.arange(256), np.bincount(X.ravel(), minlength=256), width=1.0, color="slategray")
    ax.set_yscale("log"); ax.set_xlabel("Pixel intensity (0-255)"); ax.set_ylabel("Count (log scale)")
    ax.set_title("Pixel intensity distribution (training set)")
    fig.tight_layout()
    _finish(fig, path, dpi=200)


# ----------------------------------------------------------------------------- binary task
def binary_demo(X, y_class, y_bin, pred, idx, path):
    fig, axes = plt.subplots(2, 5, figsize=(9, 4.4))
    for ax, i, p in zip(axes.ravel(), idx, pred):
        ax.imshow(X[i].reshape(28, 28), cmap="gray")
        ok = bool(p) == bool(y_bin[i])
        ax.set_title(f"{C.CLASS_NAMES[y_class[i]]}\npred: {'Shirt' if p else 'not shirt'}", fontsize=8,
                     color="green" if ok else "red")
        ax.axis("off")
    fig.tight_layout()
    _finish(fig, path, dpi=200)


def precision_recall_vs_threshold(scores, y_true, path):
    prec, rec, thr = precision_recall_curve(y_true, scores)
    lo, hi = np.percentile(scores, [1, 99.8])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(thr, prec[:-1], label="Precision")
    ax.plot(thr, rec[:-1], label="Recall")
    ax.axvline(0, color="k", linestyle=":", linewidth=1, label="Default threshold (0)")
    ax.set_xlim(lo, hi); ax.set_xlabel("Decision threshold"); ax.grid(True); ax.legend()
    ax.set_title("Precision and recall vs threshold (SGD + /255)")
    fig.tight_layout()
    _finish(fig, path, dpi=200)


def pr_roc(scores, y_true, path, title_suffix=""):
    """Precision-recall (left) and ROC (right) curves of the binary models; dots mark the default threshold."""
    n_neg = int((~y_true).sum())
    with plt.rc_context(REPORT):
        fig, (axp, axr) = plt.subplots(1, 2, figsize=(7.1, 2.6))
        for name, s in scores.items():
            p, r, _ = precision_recall_curve(y_true, s)
            f, t, _ = roc_curve(y_true, s)
            ap, auc = average_precision_score(y_true, s), roc_auc_score(y_true, s)
            lw = 1.8 if name == DIV else 1.1
            axp.plot(r, p, color=COLORS[name], lw=lw, label=f"{SHORT[name]} (AP = {ap:.3f})")
            axr.plot(f, t, color=COLORS[name], lw=lw, label=f"{SHORT[name]} (AUC = {auc:.3f})")
            pred = s >= 0
            tp, fp = int((pred & y_true).sum()), int((pred & ~y_true).sum())
            # default-threshold operating point: large marker with a white edge, drawn above the curves
            marker = dict(marker="o", ls="none", color=COLORS[name], ms=4.5, mec="white", mew=0.8, zorder=5)
            axp.plot(tp / y_true.sum(), tp / max(tp + fp, 1), **marker)
            axr.plot(fp / n_neg, tp / y_true.sum(), **marker)
        axp.axhline(y_true.mean(), color="gray", ls="--", lw=0.8, label=f"random ({y_true.mean():.2f})")
        axr.plot([0, 1], [0, 1], "k--", lw=0.8, label="random")
        axp.set_xlabel("Recall"); axp.set_ylabel("Precision"); axp.set_xlim(0, 1); axp.set_ylim(0, 1.02)
        axr.set_xlabel("False positive rate"); axr.set_ylabel("True positive rate")
        for ax in (axp, axr):
            ax.grid(alpha=0.3); ax.legend(fontsize=6)
        fig.tight_layout()
        _finish(fig, path)


def binary_confusion_pair(cms, titles, path):
    """Two 2x2 confusion matrices side by side (e.g. default threshold and the cross-validation-chosen one)."""
    fig, axes = plt.subplots(1, len(cms), figsize=(4.2 * len(cms), 3.8))
    for ax, cm, title in zip(np.atleast_1d(axes), cms, titles):
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Not shirt", "Shirt"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Not shirt", "Shirt"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=9)
    fig.tight_layout()
    _finish(fig, path, dpi=200)


# ----------------------------------------------------------------------------- multiclass task
def confusion_matrix_pct(cm, path, figsize=(3.45, 3.1)):
    """Row-normalised confusion matrix (% of each true class); cells below 0.5% are left blank."""
    cmn = cm / cm.sum(axis=1, keepdims=True)
    with plt.rc_context(REPORT):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticklabels(C.CLASS_NAMES, rotation=60, ha="right", fontsize=6)
        ax.set_yticklabels(C.CLASS_NAMES, fontsize=6)
        for i in range(10):
            for j in range(10):
                if cmn[i, j] >= 0.005:
                    ax.text(j, i, f"{cmn[i, j] * 100:.0f}", ha="center", va="center", fontsize=5.5,
                            color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
        fig.tight_layout()
        _finish(fig, path)


def per_class_f1_bars(df, path):
    """Report Fig. 4: per-class F1 of all models (double column). ``df``: classes x models."""
    with plt.rc_context({"font.size": 7.5, "font.family": "serif"}):
        fig, ax = plt.subplots(figsize=(7.2, 2.1))
        w = 0.8 / len(df.columns)
        for i, col in enumerate(df.columns):
            ax.bar(np.arange(len(df)) + (i - (len(df.columns) - 1) / 2) * w, df[col], w, label=col)
        ax.set_xticks(range(len(df))); ax.set_xticklabels(df.index)
        ax.set_ylim(0.3, 1.0); ax.set_ylabel("F1 (test set)"); ax.grid(axis="y", alpha=0.35)
        ax.legend(ncol=len(df.columns), fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False)
        fig.tight_layout()
        _finish(fig, path)
