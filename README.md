# Classifying Look-Alike Garments

**A Comparative Study of Binary, Multiclass and Multilabel Learning on Fashion-MNIST**

Course project for IAI600 (MS AI and Automation, Högskolan Väst).
Authors: Zain Imran, Ahsan Bhatti, Quratul Ain, Arsel Nazeer.

Fashion-MNIST contains 70,000 grayscale images (28 x 28) of ten garment types, several of which look almost alike (Shirt, T-shirt/top, Pullover, Coat).
We study three classification settings on it with a shared evaluation protocol (3-fold stratified cross-validation, then a single evaluation on the held-out test set):

| Task | Question | Models |
|---|---|---|
| Binary | Is this image a Shirt? (10% positives) | linear SGD classifier, three preprocessing variants |
| Multiclass | Which of the ten classes? | SGD (linear), k-nearest neighbours, random forest |
| Multilabel | footwear? top? long-sleeved? | k-nearest neighbours |

## Main results (held-out test set)

| | Result |
|---|---|
| Binary, SGD on pixels / 255 (threshold 0) | precision 0.596, recall 0.585, F1 0.590, ROC-AUC 0.913, average precision 0.632 (an always-"not shirt" baseline scores 0.90 accuracy) |
| Multiclass accuracy | SGD raw 0.809, SGD / 255 0.845, KNN 0.859, **random forest 0.885** |
| Multilabel (KNN) | macro-F1 0.959, exact-match accuracy 0.940 |

Errors concentrate in a cluster of look-alike upper-body garments (Shirt F1 0.67 for the best model). See the report for the full analysis,
including two evaluation pitfalls (the effect of standardisation, and the non-transferability of decision thresholds chosen on cross-validated scores).

## Repository structure

```
.
├── src/garments/            # the reusable code
│   ├── config.py            #   constants, class names, label definitions, paths
│   ├── data.py              #   loading, task targets, cross-validation folds
│   ├── models.py            #   model definitions (default hyper-parameters)
│   ├── evaluation.py        #   metrics, threshold analysis, error analysis
│   ├── plotting.py          #   figures (sized for the IEEE two-column report)
│   └── experiments.py       #   one function per stage of the study
├── scripts/run_experiments.py   # runs everything end to end
├── notebooks/walkthrough.ipynb  # guided walkthrough with the guiding questions and answers
├── figures/                 # generated figures
├── results/                 # generated tables (CSV) and key_results.json
├── requirements.txt
└── README.md
```

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

With Anaconda: `conda create -n garments python=3.11`, `conda activate garments`, then `pip install -r requirements.txt`.

## Data

Download the Fashion-MNIST CSV files from Kaggle
(<https://www.kaggle.com/datasets/zalando-research/fashionmnist>) and place these two files in a `data/` folder in the repository root
(the folder is not tracked by git because the training file exceeds GitHub's 100 MB limit):

```
data/fashion-mnist_train.csv
data/fashion-mnist_test.csv
```

If the files are missing, the code downloads the same dataset from OpenML instead (internet connection required). The results in the report were produced with the Kaggle files;
the OpenML copy has the same images, but we have not verified that the row order is identical, so numbers could differ slightly. To keep the data elsewhere, set the environment variable `FASHION_MNIST_DIR`.

## Running

From the repository root:

```bash
python scripts/run_experiments.py                  # full run, about 20-30 minutes on a laptop
python scripts/run_experiments.py --quick          # tiny subsample, about a minute: only checks that everything runs
python scripts/run_experiments.py --skip-convergence-check   # skips the extra SGD re-fits
```

The script prints every table and writes the tables to `results/` and the figures to `figures/`.

**In VS Code:** open the repository folder, select the Python interpreter of your environment, open `scripts/run_experiments.py` and press *Run Python File*, or open
`notebooks/walkthrough.ipynb` and *Run All* (the notebook adds the commentary and the guiding questions; `GARMENTS_QUICK=1` makes it run on the tiny subsample).
Visual Studio with the Python workload works in the same way.

## Reproducibility

* All randomness uses seed 42; all models use scikit-learn defaults (no hyper-parameter tuning).
* Cross-validation: 3-fold, stratified, shuffled; every model of a task uses identical folds (the binary task stratifies by the binary label, the ten-class and multilabel tasks by the class).
* The test set is touched only in the `*_test` functions of `src/garments/experiments.py`; model and threshold choices come from cross-validation.
* Versions used: Python 3.11, scikit-learn 1.9.1, NumPy 2.4.6, pandas 3.0.6, Matplotlib 3.11.2. Timings in the report were measured on a laptop with an Intel Core i5-1135G7 and 16 GB of RAM, without a GPU.

## Acknowledgments

We thank our instructors Eleftherios Vlahakis and Rashid Ali. Generative AI (Claude, Anthropic) assisted with project planning, writing and debugging the code, interpreting results
and drafting the report; the experiments were run by the authors. Dataset: H. Xiao, K. Rasul and R. Vollgraf, *Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms*, arXiv:1708.07747, 2017.
