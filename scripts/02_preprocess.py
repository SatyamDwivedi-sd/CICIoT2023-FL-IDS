"""
Script 02 — Data Preprocessing

Reproduces notebook 02_Preprocessing.ipynb:
  - Encodes labels (binary, 8-category, fine-34)
  - Stratified 70/10/20 train/val/test split
  - MinMaxScaler (fit on train, applied to val/test)
  - Saves all splits, scalers, encoders, and split_summary.json

Usage
-----
    python scripts/02_preprocess.py

    python scripts/02_preprocess.py \
        --clean-csv /data/extracted/ciciot_clean.csv \
        --out-dir   /data/extracted
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ciciot_ids.data.loader import RawDataLoader
from ciciot_ids.data.preprocessor import DataPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("02_preprocess")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="CICIoT2023 Preprocessing")
    parser.add_argument(
        "--clean-csv",
        type=Path,
        default=root / "extracted" / "ciciot_clean.csv",
        help="Path to ciciot_clean.csv (output of 01_eda.py)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "extracted",
        help="Directory to write all split CSV files and artifacts",
    )
    parser.add_argument("--test-size",  type=float, default=0.20)
    parser.add_argument("--val-size",   type=float, default=0.125)
    parser.add_argument("--random-seed", type=int,  default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Load clean data ────────────────────────────────────────────────────
    logger.info("Loading %s …", args.clean_csv)
    loader = RawDataLoader()
    df = loader.load_clean(args.clean_csv)
    logger.info("Loaded — shape: %s", df.shape)

    # ── 2. Build targets ──────────────────────────────────────────────────────
    preprocessor = DataPreprocessor(
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_seed,
    )
    df = preprocessor.build_targets(df)
    feature_cols = preprocessor.get_feature_columns(df)
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    X = df[feature_cols]
    y_binary  = df["target_binary"]
    y_cat8    = df["target_cat8"]
    y_fine34  = df["target_fine34"]

    # ── 3. Binary split + scale ───────────────────────────────────────────────
    logger.info("Splitting and scaling binary task …")
    (X_train_b, X_val_b, X_test_b,
     y_train_b, y_val_b, y_test_b) = preprocessor.split_and_scale_binary(X, y_binary)

    logger.info(
        "Binary — train: %s  val: %s  test: %s",
        X_train_b.shape, X_val_b.shape, X_test_b.shape,
    )

    # ── 4. Multiclass split + scale ───────────────────────────────────────────
    logger.info("Splitting and scaling multiclass (cat8 + fine34) task …")
    (X_train_m, X_val_m, X_test_m,
     y_train_cat8, y_val_cat8, y_test_cat8,
     y_train_fine34, y_val_fine34, y_test_fine34) = preprocessor.split_and_scale_cat8(
        X, y_cat8, y_fine34
    )

    logger.info(
        "Cat8 — train: %s  val: %s  test: %s",
        X_train_m.shape, X_val_m.shape, X_test_m.shape,
    )

    # ── 5. Save all artifacts ─────────────────────────────────────────────────
    logger.info("Saving artifacts to %s …", args.out_dir)
    preprocessor.save_artifacts(
        args.out_dir,
        X_train_b=X_train_b,  X_val_b=X_val_b,  X_test_b=X_test_b,
        y_train_b=y_train_b,  y_val_b=y_val_b,  y_test_b=y_test_b,
        X_train_m=X_train_m,  X_val_m=X_val_m,  X_test_m=X_test_m,
        y_train_cat8=y_train_cat8, y_val_cat8=y_val_cat8, y_test_cat8=y_test_cat8,
        y_train_fine34=y_train_fine34, y_val_fine34=y_val_fine34, y_test_fine34=y_test_fine34,
        feature_cols=feature_cols,
    )
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
