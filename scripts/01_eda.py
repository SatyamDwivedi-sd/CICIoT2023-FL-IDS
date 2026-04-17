"""
Script 01 — Exploratory Data Analysis

Reproduces notebook 01_EDA.ipynb:
  - Loads all raw CSVs from MERGED_CSV
  - Deduplicates and fixes Inf/NaN values
  - Saves ciciot_clean.csv and feature_statistics.csv
  - Saves category distribution plot

Usage (local)
-------------
    python scripts/01_eda.py

Usage (cluster / custom paths)
-------------------------------
    python scripts/01_eda.py \
        --raw-dir  /data/CICIoT2023/MERGED_CSV \
        --out-dir  /data/CICIoT2023/extracted \
        --plot-dir /data/CICIoT2023/results/eda
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Make the src package importable without installing ────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ciciot_ids.constants import CATEGORY_MAP
from ciciot_ids.data.loader import RawDataLoader
from ciciot_ids.visualization.plots import IDSPlotter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("01_eda")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="CICIoT2023 EDA")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=root / "extracted" / "MERGED_CSV",
        help="Directory containing raw per-category CSV files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "extracted",
        help="Output directory for ciciot_clean.csv and feature_statistics.csv",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=root / "results" / "eda",
        help="Directory to save EDA plots",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively (disabled by default for cluster use)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    plotter = IDSPlotter(show=args.show_plots)

    # ── 1. Load raw CSVs ──────────────────────────────────────────────────────
    logger.info("Loading raw data from %s …", args.raw_dir)
    loader = RawDataLoader()
    df = loader.load(args.raw_dir)

    logger.info("Raw shape: %s", df.shape)
    logger.info("Columns: %s", df.columns.tolist())

    # ── 2. Attach category column ─────────────────────────────────────────────
    df["Category"] = df["Label"].map(CATEGORY_MAP)
    missing = df.loc[df["Category"].isna(), "Label"].unique()
    if len(missing):
        logger.warning("Labels not in CATEGORY_MAP: %s", missing.tolist())

    logger.info("\n=== LABEL DISTRIBUTION ===\n%s", df["Label"].value_counts().to_string())
    logger.info("\n=== CATEGORY DISTRIBUTION ===\n%s", df["Category"].value_counts().to_string())

    # ── 3. Data quality checks ────────────────────────────────────────────────
    inf_counts = np.isinf(df.select_dtypes(include=np.number)).sum()
    inf_cols = inf_counts[inf_counts > 0]
    logger.info("Inf values found in %d columns", len(inf_cols))

    dupes = df.duplicated().sum()
    logger.info("Duplicate rows: %d", dupes)

    # ── 4. Clean ──────────────────────────────────────────────────────────────
    df_clean = df.drop_duplicates()
    logger.info("After dedup: %d rows (removed %d)", len(df_clean), dupes)

    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
    logger.info("After Inf/NaN fix: %d rows", len(df_clean))

    # ── 5. Feature statistics ─────────────────────────────────────────────────
    numeric_cols = df_clean.drop(columns=["Label", "Category"]).select_dtypes(include=np.number).columns
    stats = df_clean[numeric_cols].describe().T[["mean", "std", "min", "max"]]
    stats_path = args.out_dir / "feature_statistics.csv"
    stats.to_csv(stats_path)
    logger.info("Feature statistics saved → %s", stats_path)

    # ── 6. Save clean CSV ─────────────────────────────────────────────────────
    clean_path = args.out_dir / "ciciot_clean.csv"
    df_clean.to_csv(clean_path, index=False)
    logger.info("Clean data saved → %s  shape: %s", clean_path, df_clean.shape)

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    plotter.plot_class_distribution(
        counts=df_clean["Category"].value_counts(),
        title="Attack Category Distribution (after dedup)",
        save_path=args.plot_dir / "category_distribution.png",
    )

    # ── 8. Summary ────────────────────────────────────────────────────────────
    logger.info(
        "\n=== EDA COMPLETE ===\n"
        "  Original rows:     %d\n"
        "  After dedup:       %d\n"
        "  Final clean rows:  %d\n"
        "  Features:          %d\n"
        "  Attack classes:    %d\n"
        "  Categories:        %d",
        len(df),
        len(df.drop_duplicates()),
        len(df_clean),
        len(numeric_cols),
        df_clean["Label"].nunique(),
        df_clean["Category"].nunique(),
    )


if __name__ == "__main__":
    main()
