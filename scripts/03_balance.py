"""
Script 03 — Class Balancing (multiclass, cat8)

Reproduces notebook 06_Balancing_Multiclass.ipynb:
  - Applies RandomUnderSampler to majority classes
  - Applies SMOTE-ENN to minority classes
  - Saves X_train_balanced_cat8.csv / y_train_balanced_cat8.csv
  - Val and Test are intentionally left untouched (imbalanced)

Usage
-----
    python scripts/03_balance.py

    python scripts/03_balance.py \
        --data-dir   /data/extracted \
        --out-dir    /data/extracted \
        --n-jobs     8
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ciciot_ids.constants import OVER_SAMPLING_TARGETS, UNDER_SAMPLING_TARGETS
from ciciot_ids.data.balancer import ClassBalancer
from ciciot_ids.data.loader import SplitDataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("03_balance")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="CICIoT2023 Class Balancing")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "extracted",
        help="Directory containing X_train_cat8.csv and y_train_cat8.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "extracted",
        help="Directory to write balanced training set",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for SMOTE-ENN (-1 = all cores)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Load cat8 training split ───────────────────────────────────────────
    logger.info("Loading cat8 training split from %s …", args.data_dir)
    loader = SplitDataLoader()
    X_train, _, _, y_train, _, _ = loader.load_cat8(args.data_dir)

    logger.info(
        "Loaded — X: %s  y distribution: %s",
        X_train.shape,
        y_train.value_counts().sort_index().to_dict(),
    )

    # ── 2. Balance ────────────────────────────────────────────────────────────
    balancer = ClassBalancer(
        under_targets=UNDER_SAMPLING_TARGETS,
        over_targets=OVER_SAMPLING_TARGETS,
        random_state=args.random_seed,
        n_jobs=args.n_jobs,
    )
    X_bal, y_bal = balancer.balance(X_train, y_train)

    logger.info(
        "Balanced — X: %s  y distribution: %s",
        X_bal.shape,
        y_bal.value_counts().sort_index().to_dict(),
    )

    # ── 3. Save ───────────────────────────────────────────────────────────────
    balancer.save_balanced(
        args.out_dir,
        X_bal,
        y_bal,
        prefix="balanced_cat8",
    )
    logger.info("Balancing complete.")


if __name__ == "__main__":
    main()
