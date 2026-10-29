"""The experiments of the project, one function per stage.

Each function computes its results, writes tables to ``results/`` and figures to ``figures/``, and returns the
results as a dict, so the same code drives both ``scripts/run_experiments.py`` and the walkthrough notebook.
Model choices are made with cross-validation on the training set only; the ``*_test`` functions are the single
final evaluation on the held-out test set.
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import SGDClassifier

from . import config as C
from . import plotting as P
from .data import binary_target, cv_splits, multilabel_target
from .evaluation import (accuracy_ci95, binary_confusion, binary_metrics, class_destinations, classes_to_labels,
                         exact_match_by_class, fold_accuracies, multilabel_scores, per_class_f1, per_label_report,
                         threshold_policies, threshold_table, timed, top_confusions, top_k_class_mix)
from .models import (DIV, MAIN_BINARY, RAW, STD, binary_models, div255, final_estimator, multiclass_models,
                     multilabel_model)


def _save(df, name):
    C.ensure_dirs()
    df.to_csv(C.RES_DIR / name)


def _fig(name):
    C.ensure_dirs()
    return C.FIG_DIR / name


# =============================================================================================== exploration
def eda(data):
    """Dataset facts, class balance, pixel statistics, duplicates, and exploratory figures."""
    X, y, Xt, yt = data.X_train, data.y_train, data.X_test, data.y_test
    counts = pd.DataFrame({"label": range(10), "class": C.CLASS_NAMES,
                           "train": np.bincount(y, minlength=10), "test": np.bincount(yt, minlength=10)})
    counts["train_%"] = (100 * counts["train"] / counts["train"].sum()).round(1)

    train_rows = {r.tobytes() for r in X}
    stats = {
        "data source": data.source,
        "images (train + test)": len(X) + len(Xt), "train images": len(X), "test images": len(Xt),
        "features per image": X.shape[1], "classes": int(len(np.unique(y))),
        "pixel range": f"{int(X.min())}-{int(X.max())}",
        "mean pixel intensity": round(float(X.mean()), 1), "std of pixel values": round(float(X.std()), 1),
        "pixels equal to 0": f"{(X == 0).mean():.1%}",
        "duplicate images within train": int(len(X) - len(np.unique(X, axis=0))),
        "test images also present in train": int(sum(r.tobytes() in train_rows for r in Xt)),
    }
    _save(counts.set_index("label"), "class_counts.csv")
    P.class_distribution(y, _fig("eda_class_distribution.png"))
    P.sample_grid(X, y, _fig("eda_sample_grid.png"))
    P.mean_images(X, y, _fig("fig_mean_images.png"))
    P.pixel_histogram(X, _fig("eda_pixel_histogram.png"))
    return {"class_counts": counts, "stats": stats}


# =============================================================================================== binary task
def binary_demo(data):
    """Qualitative check: raw-pixel SGD on 10 hand-picked training images (5 shirts, 4 look-alikes, 1 trouser)."""
    X, y = data.X_train, data.y_train
    y_bin = binary_target(y)
    model = binary_models()[RAW]
    model.fit(X, y_bin)
    rng = np.random.default_rng(C.RANDOM_STATE)
    shirts = rng.choice(np.flatnonzero(y == C.POSITIVE_CLASS), 5, replace=False)
    others = [rng.choice(np.flatnonzero(y == c)) for c in (0, 2, 4, 3)] + [rng.choice(np.flatnonzero(y == 1))]
    idx = np.concatenate([shirts, others])
    pred = model.predict(X[idx])
    table = pd.DataFrame({"index": idx, "true class": [C.CLASS_NAMES[y[i]] for i in idx],
                          "is shirt": y_bin[idx], "predicted shirt": pred})
    table["correct"] = table["is shirt"] == table["predicted shirt"]
    P.binary_demo(X, y, y_bin, pred, idx, _fig("binary_demo_predictions.png"))
    return table


def binary_cv(data, check_convergence=True):
    """Shirt-vs-rest with 3-fold cross-validation: three preprocessing variants, thresholds, diagnostics."""
    X, y = data.X_train, data.y_train
    y_bin = binary_target(y)
    splits = cv_splits(X, y_bin)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        scores = {name: cross_val_predict(m, X, y_bin, cv=splits, method="decision_function")
                  for name, m in binary_models().items()}

    rows = {}
    for name, s in scores.items():
        m = binary_metrics(s, y_bin, 0.0)
        pred = s >= 0
        m.update({"ROC-AUC": roc_auc_score(y_bin, s), "average precision": average_precision_score(y_bin, s),
                  "fold accuracies": " / ".join(f"{a:.4f}" for a in fold_accuracies(y_bin, pred, splits))})
        rows[name] = m
    rows["Dummy (always 'not shirt')"] = {
        "threshold": None, "% flagged": 0.0, "accuracy": 1 - y_bin.mean(), "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "TP": 0, "FP": 0, "FN": int(y_bin.sum()), "TN": int((~y_bin).sum()), "ROC-AUC": 0.5,
        "average precision": y_bin.mean(), "fold accuracies": "-"}
    cv_table = pd.DataFrame(rows).T
    _save(cv_table, "binary_cv_metrics.csv")

    s_main = scores[MAIN_BINARY]
    thr_table = threshold_table(s_main, y_bin)
    policies = threshold_policies(s_main, y_bin)
    thr_maxf1 = float(policies.loc[policies["policy"] == "max F1", "threshold"].iloc[0])
    _save(thr_table.set_index("threshold"), "binary_cv_threshold_table.csv")
    _save(policies.set_index("policy"), "binary_cv_threshold_policies.csv")

    top_mix = pd.DataFrame({name: top_k_class_mix(s, y) for name, s in scores.items()}).fillna(0).astype(int)
    top_mix = top_mix.loc[top_mix.sum(axis=1).sort_values(ascending=False).index]
    top_precision = {name: top_mix.loc["Shirt", name] / 600 for name in top_mix.columns}

    P.precision_recall_vs_threshold(s_main, y_bin, _fig("binary_precision_recall_vs_threshold_cv.png"))
    P.pr_roc(scores, y_bin, _fig("binary_pr_roc_cv.png"))

    fp_mask = (~y_bin) & (s_main >= 0)
    fp_by_class = (pd.Series(y[fp_mask]).map(dict(enumerate(C.CLASS_NAMES))).value_counts()
                   .rename("false positives").to_frame())
    class_size = pd.Series(np.bincount(y, minlength=10), index=C.CLASS_NAMES)
    fp_by_class["% of that class's images"] = (100 * fp_by_class["false positives"]
                                               / class_size.loc[fp_by_class.index]).round(1)

    out = {"y_bin": y_bin, "splits": splits, "scores": scores, "cv_table": cv_table, "thr_table": thr_table,
           "policies": policies, "thr_maxf1": thr_maxf1, "top_mix": top_mix, "top_precision": top_precision,
           "fp_by_class": fp_by_class, "flagged_cv": {"threshold 0": float((s_main >= 0).mean()),
                                                      "max-F1 threshold": float((s_main >= thr_maxf1).mean())}}
    if check_convergence:
        out["convergence"] = _convergence_check(X, y_bin, splits, scores[STD])
        _save(out["convergence"]["fits"].set_index(["model", "max_iter"]), "binary_convergence_check.csv")
    return out


def _convergence_check(X, y_bin, splits, scores_std_1000):
    """Does the SGD optimiser converge, and does allowing more iterations change the conclusions?"""
    fits = []
    for name in (STD, DIV):
        for max_iter in (1000, 5000):
            model = binary_models(max_iter=max_iter)[name]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(X, y_bin)
            fits.append({"model": name, "max_iter": max_iter, "iterations used": int(final_estimator(model).n_iter_),
                         "ConvergenceWarning": any(issubclass(w.category, ConvergenceWarning) for w in caught)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        s5k = cross_val_predict(binary_models(max_iter=5000)[STD], X, y_bin, cv=splits, method="decision_function")
    auc = {"max_iter=1000": roc_auc_score(y_bin, scores_std_1000), "max_iter=5000": roc_auc_score(y_bin, s5k)}
    return {"fits": pd.DataFrame(fits), "standardised ROC-AUC": auc}


def binary_test(data, bcv):
    """Final evaluation of the binary models on the test set (decisions fixed beforehand by cross-validation)."""
    X, y, Xt, yt = data.X_train, data.y_train, data.X_test, data.y_test
    y_bin, yt_bin = binary_target(y), binary_target(yt)
    scores, fit_s = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for name, m in binary_models().items():
            _, fit_s[name] = timed(m.fit, X, y_bin)
            scores[name] = m.decision_function(Xt)

    def row(s, thr):
        m = binary_metrics(s, yt_bin, thr)
        m.update({"ROC-AUC": roc_auc_score(yt_bin, s), "average precision": average_precision_score(yt_bin, s)})
        return m

    thr = bcv["thr_maxf1"]
    rows = {f"{n} @ threshold 0": row(s, 0.0) for n, s in scores.items()}
    rows[f"{MAIN_BINARY} @ threshold {thr:.3f} (max F1 in CV)"] = row(scores[MAIN_BINARY], thr)
    rows["Dummy (always 'not shirt')"] = {
        "threshold": None, "% flagged": 0.0, "accuracy": 1 - yt_bin.mean(), "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "TP": 0, "FP": 0, "FN": int(yt_bin.sum()), "TN": int((~yt_bin).sum()), "ROC-AUC": 0.5,
        "average precision": yt_bin.mean()}
    table = pd.DataFrame(rows).T
    _save(table, "final_binary_test_results.csv")

    P.pr_roc(scores, yt_bin, _fig("fig_pr_roc.png"))
    cms = []
    for t in (0.0, thr):
        tp, fp, fn, tn = binary_confusion(scores[MAIN_BINARY], yt_bin, t)
        cms.append(np.array([[tn, fp], [fn, tp]]))
    P.binary_confusion_pair(cms, [f"{MAIN_BINARY}, threshold 0", f"{MAIN_BINARY}, threshold {thr:.2f} (max F1)"],
                            _fig("final_binary_confusion_matrices_test.png"))
    flagged_test = {"threshold 0": float((scores[MAIN_BINARY] >= 0).mean()),
                    "max-F1 threshold": float((scores[MAIN_BINARY] >= thr).mean())}
    return {"table": table, "scores": scores, "fit_seconds": fit_s, "flagged_test": flagged_test}


# =============================================================================================== multiclass task
def multiclass_mechanics(data):
    """How scikit-learn handles 10 classes: decision scores for one image, one-vs-rest versus one-vs-one."""
    X, y = data.X_train, data.y_train
    first = X[:1]
    out = {"true class of first image": C.CLASS_NAMES[y[0]]}
    for name, model in list(multiclass_models().items())[:2]:
        model.fit(X, y)
        est = final_estimator(model)
        s = model.decision_function(first)[0]
        out[name] = {
            "coef_ shape": est.coef_.shape,
            "scores": pd.DataFrame({"class": C.CLASS_NAMES, "decision score": s})
                        .sort_values("decision score", ascending=False).round(3).reset_index(drop=True),
            "predicted": C.CLASS_NAMES[int(model.predict(first)[0])],
        }
    ovo = OneVsOneClassifier(make_pipeline(FunctionTransformer(div255), SGDClassifier(random_state=C.RANDOM_STATE)))
    ovo.fit(X, y)
    out["one-vs-one"] = {"binary classifiers": len(ovo.estimators_), "predicted": C.CLASS_NAMES[int(ovo.predict(first)[0])]}
    return out


def multiclass_cv(data):
    """Ten classes, four models, 3-fold cross-validation."""
    X, y = data.X_train, data.y_train
    splits = cv_splits(X, y)
    preds, scores, seconds = {}, {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for name, m in multiclass_models().items():
            if name.startswith("SGD"):
                s, t = timed(cross_val_predict, m, X, y, cv=splits, method="decision_function")
                scores[name], preds[name] = s, s.argmax(axis=1).astype(np.uint8)
            else:
                p, t = timed(cross_val_predict, m, X, y, cv=splits)
                preds[name] = p.astype(np.uint8)
            seconds[name] = t

    summary = pd.DataFrame({
        name: {"accuracy": (p == y).mean(), "macro-F1": f1_score(y, p, average="macro"),
               "Shirt F1": f1_score(y, p, labels=[C.POSITIVE_CLASS], average=None)[0],
               "fold accuracies": " / ".join(f"{a:.4f}" for a in fold_accuracies(y, p, splits)),
               "CV time (s)": round(seconds[name], 1)}
        for name, p in preds.items()}).T
    dummy = np.zeros_like(y)
    summary.loc["Dummy (always one class)"] = {"accuracy": (dummy == y).mean(),
                                               "macro-F1": f1_score(y, dummy, average="macro", zero_division=0),
                                               "Shirt F1": 0.0, "fold accuracies": "-", "CV time (s)": 0.0}
    _save(summary, "multiclass_cv_summary.csv")

    best = summary.loc[list(preds), "accuracy"].astype(float).idxmax()      # chosen by cross-validation
    cms = {name: confusion_matrix(y, p) for name, p in preds.items()}
    f1_table = pd.DataFrame({name: per_class_f1(y, p) for name, p in preds.items()}, index=C.CLASS_NAMES)
    _save(f1_table.round(4), "multiclass_cv_per_class_f1.csv")
    P.confusion_matrix_pct(cms[best], _fig("multiclass_confusion_matrix_cv_best.png"))
    return {"splits": splits, "preds": preds, "scores": scores, "summary": summary, "best": best, "cms": cms,
            "per_class_f1": f1_table, "top_confusions": top_confusions(cms[best]),
            "shirt_destinations": class_destinations(cms[best]),
            "top_confusions_sgd": top_confusions(cms["SGD + /255"]),
            "shirt_destinations_sgd": class_destinations(cms["SGD + /255"])}


def multiclass_test(data, mcv):
    """Final ten-class evaluation: every model fitted once on all training data and scored on the test set."""
    X, y, Xt, yt = data.X_train, data.y_train, data.X_test, data.y_test
    preds, fit_s, pred_s = {}, {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for name, m in multiclass_models().items():
            _, fit_s[name] = timed(m.fit, X, y)
            preds[name], pred_s[name] = timed(m.predict, Xt)
    best = mcv["best"]                                  # chosen by cross-validation, not by test results
    rows = {}
    for name, p in preds.items():
        acc = float((p == yt).mean())
        rows[name] = {"test accuracy": acc, "95% CI (+/-)": accuracy_ci95(acc, len(yt)),
                      "test macro-F1": f1_score(yt, p, average="macro"),
                      "test Shirt F1": f1_score(yt, p, labels=[C.POSITIVE_CLASS], average=None)[0],
                      "CV accuracy": float(mcv["summary"].loc[name, "accuracy"]),
                      "fit time (s)": round(fit_s[name], 1), "predict time (s)": round(pred_s[name], 1)}
    table = pd.DataFrame(rows).T
    _save(table.round(4), "final_multiclass_test_results.csv")

    f1_table = pd.DataFrame({name: per_class_f1(yt, p) for name, p in preds.items()}, index=C.CLASS_NAMES)
    _save(f1_table.round(4), "final_multiclass_per_class_f1_test.csv")
    cm = confusion_matrix(yt, preds[best])
    P.confusion_matrix_pct(cm, _fig("fig_confusion_best_model.png"))
    P.per_class_f1_bars(f1_table.rename(columns={"SGD, raw pixels": "SGD, raw pixels", "SGD + /255": "SGD, \u00f7255"}),
                        _fig("fig_per_class_f1.png"))
    return {"preds": preds, "table": table, "per_class_f1": f1_table, "best": best,
            "shirt_destinations": class_destinations(cm), "top_confusions": top_confusions(cm)}


# =============================================================================================== multilabel task
def multilabel_demo(data):
    """True and predicted label vector of the first training image."""
    X, y = data.X_train, data.y_train
    Y = multilabel_target(y)
    model = multilabel_model().fit(X, Y)
    pred = model.predict(X[:1])[0].astype(int)
    return {"true class": C.CLASS_NAMES[y[0]], "true labels": dict(zip(C.LABEL_DEFS, Y[0].astype(int))),
            "predicted labels": dict(zip(C.LABEL_DEFS, pred))}


def label_summary(data):
    """Definition, prevalence and co-occurrence of the three labels."""
    y = data.y_train
    Y = multilabel_target(y)
    defs = pd.DataFrame({"label": list(C.LABEL_DEFS),
                         "classes": [", ".join(C.CLASS_NAMES[c] for c in cls) for cls in C.LABEL_DEFS.values()],
                         "prevalence": Y.mean(axis=0).round(3)})
    active = pd.Series(Y.sum(axis=1)).value_counts().sort_index().rename("images").to_frame()
    active["share"] = (active["images"] / len(Y)).round(3)
    active.index.name = "labels active per image"
    return {"definitions": defs, "labels active": active}


def multilabel_cv(data, mcv):
    """Multilabel KNN with 3-fold cross-validation (same folds as the ten-class task)."""
    X, y = data.X_train, data.y_train
    Y = multilabel_target(y)
    pred = cross_val_predict(multilabel_model(), X, Y, cv=mcv["splits"]).astype(bool)
    scores = pd.DataFrame({"KNN (k=5)": multilabel_scores(Y, pred),
                           "Baseline: all labels = yes": multilabel_scores(Y, np.ones_like(Y))}).T
    route = pd.DataFrame({
        "KNN multilabel (direct)": multilabel_scores(Y, pred),
        "SGD multiclass /255, mapped to labels": multilabel_scores(Y, classes_to_labels(mcv["preds"]["SGD + /255"])),
    }).T
    _save(scores.round(4), "multilabel_cv_summary.csv")
    _save(route.round(4), "multilabel_cv_route_comparison.csv")
    return {"scores": scores, "per_label": per_label_report(Y, pred), "exact_by_class": exact_match_by_class(y, Y, pred),
            "route_comparison": route,
            "fold_macro_f1": [f1_score(Y[va], pred[va], average="macro") for _, va in mcv["splits"]]}


def multilabel_test(data):
    """Final multilabel evaluation on the test set."""
    X, y, Xt, yt = data.X_train, data.y_train, data.X_test, data.y_test
    Y, Yt = multilabel_target(y), multilabel_target(yt)
    model = multilabel_model()
    _, fit_s = timed(model.fit, X, Y)
    pred, pred_s = timed(model.predict, Xt)
    pred = pred.astype(bool)
    scores = pd.DataFrame({"KNN (k=5)": multilabel_scores(Yt, pred),
                           "Baseline: all labels = yes": multilabel_scores(Yt, np.ones_like(Yt))}).T
    _save(scores.round(4), "final_multilabel_test_results.csv")
    per_label = per_label_report(Yt, pred)
    _save(per_label, "final_multilabel_per_label_test.csv")
    return {"scores": scores, "per_label": per_label, "fit_seconds": fit_s, "predict_seconds": pred_s}


# =============================================================================================== summary
def key_results(bcv, btest, mcv, mtest, mlcv, mltest):
    """The numbers quoted in the report, collected in one place (also written to results/key_results.json)."""
    main_row = f"{MAIN_BINARY} @ threshold 0"
    out = {
        "binary_cv_main": {k: float(bcv["cv_table"].loc[MAIN_BINARY, k]) for k in
                           ("accuracy", "precision", "recall", "f1", "ROC-AUC", "average precision")},
        "binary_test_main": {k: float(btest["table"].loc[main_row, k]) for k in
                             ("accuracy", "precision", "recall", "f1", "ROC-AUC", "average precision")},
        "binary_cv_threshold_max_f1": bcv["thr_maxf1"],
        "binary_flagged_fraction": {"cv": bcv["flagged_cv"], "test": btest["flagged_test"]},
        "multiclass_cv_accuracy": {n: float(mcv["summary"].loc[n, "accuracy"]) for n in mtest["table"].index},
        "multiclass_test_accuracy": {n: float(v) for n, v in mtest["table"]["test accuracy"].items()},
        "multiclass_best_by_cv": mcv["best"],
        "multilabel_cv_macro_f1": float(mlcv["scores"].loc["KNN (k=5)", "macro-F1"]),
        "multilabel_test_macro_f1": float(mltest["scores"].loc["KNN (k=5)", "macro-F1"]),
        "multilabel_test_exact_match": float(mltest["scores"].loc["KNN (k=5)", "exact-match accuracy"]),
    }
    C.ensure_dirs()
    with open(C.RES_DIR / "key_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out
