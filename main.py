"""
main.py — command-line entry point.

Version: 1.0.0
Changelog:
  1.0.0 (2026-10-10) — eda / run / both commands, synthetic demo switch,
                       model / sampling / protocol / n-iter options.

Examples
--------
  python main.py eda  --data data/credit_card.csv
  python main.py run  --data data/credit_card.csv --n-iter 12
  python main.py run  --data data/credit_card.csv --models knn prob_knn xgboost \
                      --samplings smote adasyn --protocol paper
  python main.py all  --synthetic --n-iter 3          # quick demo, no CSV needed
"""

__version__ = "1.0.0"

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from src.data_loader import load_csv, make_synthetic, prepare_frame  # noqa: E402
from src.eda import run_eda  # noqa: E402
from src.pipeline import run_study  # noqa: E402


def load_data(args):
    if args.synthetic:
        print("Using SYNTHETIC demo data (schema mimics the reference dataset).")
        return (*prepare_frame(make_synthetic(), config.TARGET_COLUMN), True)
    if not args.data:
        sys.exit("Provide --data path/to/file.csv (or --synthetic for a demo).")
    return (*load_csv(args.data, args.target), False)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Credit-card fraud detection: reproduction "
                                             "+ hyperparameter optimisation")
    ap.add_argument("command", choices=["eda", "run", "all"])
    ap.add_argument("--data", help="CSV file with the dataset")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic demo data")
    ap.add_argument("--target", default=config.TARGET_COLUMN)
    ap.add_argument("--models", nargs="+", default=config.PAPER_MODELS,
                    choices=config.ALL_MODELS)
    ap.add_argument("--samplings", nargs="+", default=config.SAMPLINGS,
                    choices=config.SAMPLINGS)
    ap.add_argument("--protocol", default=config.DEFAULT_PROTOCOL, choices=["strict", "paper"])
    ap.add_argument("--n-iter", type=int, default=config.DEFAULT_N_ITER)
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--out", default=config.RESULTS_DIR)
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    X, y, info, synthetic = load_data(args)
    print(f"Loaded {info['rows']:,} rows, {info['features']} features, "
          f"{info['fraud']} fraud ({info['fraud_rate']:.2%}).")

    if args.command in ("eda", "all"):
        print("\nRunning EDA ...")
        paths, _ = run_eda(X, y, os.path.join(args.out, "eda"))
        print(f"EDA: {len(paths)} figures in {os.path.join(args.out, 'eda')}")
    if args.command in ("run", "all"):
        run_study(X, y, info, models=args.models, samplings=args.samplings,
                  protocol=args.protocol, n_iter=args.n_iter,
                  do_sensitivity=not args.no_sensitivity, out_dir=args.out,
                  synthetic=synthetic)


if __name__ == "__main__":
    main()
