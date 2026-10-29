"""Model definitions. All hyper-parameters are scikit-learn defaults (no tuning)."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .config import RANDOM_STATE

RAW, STD, DIV = "SGD, raw pixels", "SGD + StandardScaler", "SGD + /255"
MAIN_BINARY = DIV                      # main linear model


def div255(X):
    """Scale pixels to [0, 1] with one global constant."""
    return X / 255.0


def _sgd(max_iter=None, **kw):
    if max_iter is not None:
        kw["max_iter"] = max_iter
    return SGDClassifier(random_state=RANDOM_STATE, **kw)


def binary_models(max_iter=None):
    return {
        RAW: _sgd(max_iter),
        STD: make_pipeline(StandardScaler(), _sgd(max_iter)),
        DIV: make_pipeline(FunctionTransformer(div255), _sgd(max_iter)),
    }


def multiclass_models():
    # n_jobs=-1 trains the 10 one-vs-rest binary classifiers in parallel (results are identical to n_jobs=1)
    return {
        "SGD, raw pixels": _sgd(n_jobs=-1),
        "SGD + /255": make_pipeline(FunctionTransformer(div255), _sgd(n_jobs=-1)),
        "KNN (k=5)": KNeighborsClassifier(n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    }


def multilabel_model():
    return KNeighborsClassifier(n_jobs=-1)


def final_estimator(model):
    """The last step of a pipeline, or the model itself."""
    return model[-1] if hasattr(model, "steps") else model
