"""Data loading, task targets and cross-validation folds."""
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config as C

Data = namedtuple("Data", ["X_train", "y_train", "X_test", "y_test", "source"])


def _read_csv(path):
    df = pd.read_csv(path)
    return df.drop(columns="label").to_numpy(dtype=np.uint8), df["label"].to_numpy(dtype=np.uint8)


def _stratified_subsample(X, y, n_per_class, seed=C.RANDOM_STATE):
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), n_per_class, replace=False) for c in range(10)])
    idx = rng.permutation(idx)
    return X[idx], y[idx]


def load_data(data_dir=None, quick=False):
    """Load Fashion-MNIST.

    Uses the Kaggle CSV files (fashion-mnist_train.csv / fashion-mnist_test.csv) in ``data_dir``
    if present; otherwise downloads the same dataset from OpenML. With ``quick=True`` a small
    stratified subsample is used (for a fast smoke test, not for reporting results).
    """
    data_dir = C.DATA_DIR if data_dir is None else data_dir
    train_csv, test_csv = data_dir / "fashion-mnist_train.csv", data_dir / "fashion-mnist_test.csv"
    if train_csv.exists() and test_csv.exists():
        X_train, y_train = _read_csv(train_csv)
        X_test, y_test = _read_csv(test_csv)
        source = "Kaggle CSV"
    else:
        from sklearn.datasets import fetch_openml
        fm = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
        X, y = fm["data"].astype(np.uint8), fm["target"].astype(np.uint8)
        X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
        source = "OpenML (Fashion-MNIST, version 1)"

    if quick:
        X_train, y_train = _stratified_subsample(X_train, y_train, 200)
        X_test, y_test = _stratified_subsample(X_test, y_test, 50)
        source += ", QUICK subsample"
    else:
        assert X_train.shape == (60000, 784) and X_test.shape == (10000, 784), "unexpected dataset shape"
    return Data(X_train, y_train, X_test, y_test, source)


def binary_target(y):
    """True for the positive class (Shirt)."""
    return y == C.POSITIVE_CLASS


def multilabel_target(y):
    """Boolean matrix (n_samples, 3): footwear, top, long-sleeved."""
    return np.c_[[np.isin(y, cls) for cls in C.LABEL_DEFS.values()]].T


def cv_splits(X, y):
    """Stratified, shuffled 3-fold splits (fixed seed); the same ``y`` always gives the same folds."""
    return list(StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.RANDOM_STATE).split(X, y))
