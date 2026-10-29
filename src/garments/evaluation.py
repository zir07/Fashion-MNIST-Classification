"""Metrics, threshold analysis and error-analysis helpers."""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, f1_score, hamming_loss,
                             precision_recall_curve)

from . import config as C


def timed(fn, *args, **kwargs):
    """Run ``fn`` and return (result, elapsed seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def accuracy_ci95(acc, n):
    """Half-width of the normal-approximation 95% interval for an accuracy measured on n samples."""
    return 1.96 * np.sqrt(acc * (1 - acc) / n)


def fold_accuracies(y_true, y_pred, splits):
    return [accuracy_score(y_true[va], y_pred[va]) for _, va in splits]


# ----------------------------------------------------------------------------- binary task
def binary_confusion(scores, y_true, thr=0.0):
    pred = scores >= thr
    tp = int((pred & y_true).sum())
    fp = int((pred & ~y_true).sum())
    fn = int((~pred & y_true).sum())
    tn = int((~pred & ~y_true).sum())
    return tp, fp, fn, tn


def binary_metrics(scores, y_true, thr=0.0):
    tp, fp, fn, tn = binary_confusion(scores, y_true, thr)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": thr, "% flagged": 100 * (tp + fp) / len(y_true), "accuracy": (tp + tn) / len(y_true),
            "precision": prec, "recall": rec, "f1": f1, "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def threshold_table(scores, y_true):
    """Precision/recall at the default threshold 0 and at score quantiles (flagging the top 50%, 30%, ... 1%)."""
    thrs = sorted(set([0.0] + list(np.percentile(scores, [50, 70, 80, 90, 95, 98, 99]))))
    tab = pd.DataFrame([binary_metrics(scores, y_true, t) for t in thrs])
    tab["note"] = np.where(tab["threshold"] == 0.0, "default (0)", "")
    return tab


def threshold_policies(scores, y_true):
    """Thresholds chosen by different policies: precision target, recall target, maximum F1."""
    prec, rec, thr = precision_recall_curve(y_true, scores)

    def row(policy, i):
        p, r = prec[i], rec[i]
        return {"policy": policy, "threshold": thr[i], "precision": p, "recall": r,
                "f1": 2 * p * r / (p + r) if p + r else 0.0}

    rows = []
    for t in (0.7, 0.8, 0.9):                       # lowest threshold reaching the precision target
        ok = np.flatnonzero(prec[:-1] >= t)
        rows.append(row(f"precision >= {t:.0%}", ok[0]) if len(ok) else
                    {"policy": f"precision >= {t:.0%}", "threshold": np.nan, "precision": np.nan,
                     "recall": np.nan, "f1": np.nan})
    for t in (0.6, 0.7, 0.8):                       # highest threshold keeping the recall target
        ok = np.flatnonzero(rec[:-1] >= t)
        rows.append(row(f"recall >= {t:.0%}", ok[-1]) if len(ok) else
                    {"policy": f"recall >= {t:.0%}", "threshold": np.nan, "precision": np.nan,
                     "recall": np.nan, "f1": np.nan})
    f1c = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    rows.append(row("max F1", int(np.argmax(f1c))))
    return pd.DataFrame(rows)


def top_k_class_mix(scores, y_class, k=600):
    """True classes of the k highest-scored images (a healthy model's top scores are almost all positives)."""
    top = np.argsort(scores)[::-1][:k]
    return pd.Series(y_class[top]).map(dict(enumerate(C.CLASS_NAMES))).value_counts()


# ----------------------------------------------------------------------------- multiclass task
def per_class_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None)


def top_confusions(cm, k=10):
    rows = [(C.CLASS_NAMES[i], C.CLASS_NAMES[j], int(cm[i, j]), round(100 * cm[i, j] / cm[i].sum(), 1))
            for i in range(len(cm)) for j in range(len(cm)) if i != j]
    return (pd.DataFrame(rows, columns=["true class", "predicted as", "count", "% of true class"])
            .sort_values("count", ascending=False).head(k).reset_index(drop=True))


def class_destinations(cm, true_class=C.POSITIVE_CLASS):
    """Where do the images of one true class end up (% per predicted class)?"""
    pct = cm[true_class] / cm[true_class].sum() * 100
    return pd.Series(pct, index=C.CLASS_NAMES).round(1).sort_values(ascending=False).rename("% of images")


# ----------------------------------------------------------------------------- multilabel task
def multilabel_scores(y_true, y_pred):
    return {
        "macro-F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro-F1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "weighted-F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        # zero_division=1: an image with no true label (Trouser, Dress, Bag) that is correctly predicted as
        # "no label" counts as a perfect 1.0 (scikit-learn's default would count it as 0)
        "samples-F1": f1_score(y_true, y_pred, average="samples", zero_division=1),
        "exact-match accuracy": accuracy_score(y_true, y_pred),
        "Hamming loss": hamming_loss(y_true, y_pred),
    }


def per_label_report(y_true, y_pred):
    rep = pd.DataFrame(classification_report(y_true, y_pred, target_names=list(C.LABEL_DEFS),
                                             output_dict=True, zero_division=0)).T
    return rep.drop(index="samples avg", errors="ignore").round(4)


def exact_match_by_class(y_class, y_true, y_pred):
    exact = (y_pred == y_true).all(axis=1)
    return (pd.Series(exact).groupby(pd.Series(y_class).map(dict(enumerate(C.CLASS_NAMES)))).mean()
            .rename("exactly right").sort_values().round(4))


def classes_to_labels(class_pred):
    """Map class predictions to the three multilabel targets."""
    return np.c_[[np.isin(class_pred, cls) for cls in C.LABEL_DEFS.values()]].T
