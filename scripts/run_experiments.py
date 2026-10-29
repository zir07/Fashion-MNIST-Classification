"""Run the complete project end to end and write all tables to results/ and figures to figures/.

Usage (from the repository root):
    python scripts/run_experiments.py            # full run (about 20-30 minutes on a laptop)
    python scripts/run_experiments.py --quick    # tiny subsample, about a minute; only checks that the code runs
    python scripts/run_experiments.py --skip-convergence-check

The Kaggle CSV files go in data/ (see README). Without them the dataset is downloaded from OpenML.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from garments import config as C
from garments import experiments as E
from garments.data import load_data

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def stage(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="use a tiny subsample (smoke test; results are not meaningful)")
    parser.add_argument("--skip-convergence-check", action="store_true", help="skip the SGD convergence re-fits (saves a few minutes)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    C.ensure_dirs()
    data = load_data(quick=args.quick)
    print(f"Data source: {data.source} | train {data.X_train.shape} | test {data.X_test.shape}")

    stage("1. Exploratory data analysis")
    eda = E.eda(data)
    print(pd.Series(eda["stats"]).to_string())

    stage("2. Binary task (Shirt vs rest): cross-validation")
    bcv = E.binary_cv(data, check_convergence=not args.skip_convergence_check)
    print(bcv["cv_table"].drop(columns=["% flagged"]).round(4).to_string())
    print("\nThreshold policies (SGD + /255, cross-validated scores):\n", bcv["policies"].round(4).to_string())
    print("\nPrecision among the 600 highest scores:", {k: round(float(v), 3) for k, v in bcv["top_precision"].items()})
    if "convergence" in bcv:
        print("\nConvergence check:\n", bcv["convergence"]["fits"].to_string(),
              "\nStandardised model ROC-AUC:", bcv["convergence"]["standardised ROC-AUC"])

    stage("3. Multiclass task (10 classes): cross-validation of four models")
    mcv = E.multiclass_cv(data)
    print(mcv["summary"].to_string())
    print("\nBest model by cross-validated accuracy:", mcv["best"])
    print("\nMost frequent confusions (best model):\n", mcv["top_confusions"].to_string())
    print("\nWhere do the true Shirts go (best model)?\n", mcv["shirt_destinations"].to_string())

    stage("4. Multilabel task: cross-validation")
    mlcv = E.multilabel_cv(data, mcv)
    print(mlcv["scores"].round(4).to_string())
    print("\n", mlcv["per_label"].to_string())
    print("\nRoute comparison:\n", mlcv["route_comparison"].round(4).to_string())

    stage("5. FINAL EVALUATION on the held-out test set (decisions fixed by cross-validation)")
    btest = E.binary_test(data, bcv)
    print("Binary task:\n", btest["table"].round(4).to_string())
    mtest = E.multiclass_test(data, mcv)
    print("\nMulticlass task:\n", mtest["table"].round(4).to_string())
    print("\nWhere do the true Shirts go on the test set (", mtest["best"], ")?\n", mtest["shirt_destinations"].to_string())
    mltest = E.multilabel_test(data)
    print("\nMultilabel task:\n", mltest["scores"].round(4).to_string(), "\n", mltest["per_label"].to_string())

    stage("6. Key results quoted in the report")
    key = E.key_results(bcv, btest, mcv, mtest, mlcv, mltest)
    for k, v in key.items():
        print(f"{k}: {v}")
    print(f"\nFinished in {(time.perf_counter() - t0) / 60:.1f} minutes. Tables are in results/, figures in figures/.")


if __name__ == "__main__":
    main()
