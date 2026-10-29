"""Project-wide constants and paths."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("FASHION_MNIST_DIR", ROOT / "data"))
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"

RANDOM_STATE = 42
N_FOLDS = 3

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

POSITIVE_CLASS = 6          # "Shirt" is the positive class of the binary task

# Multilabel task: three independent yes/no labels derived from the class
LABEL_DEFS = {
    "footwear": [5, 7, 9],          # Sandal, Sneaker, Ankle boot
    "top": [0, 2, 4, 6],            # T-shirt/top, Pullover, Coat, Shirt
    "long-sleeved": [2, 4, 6],      # Pullover, Coat, Shirt
}


def ensure_dirs():
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
