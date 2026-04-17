"""
Data loading layer (Single Responsibility: only reads files, no transforms).

Two loaders:
  RawDataLoader  — merges raw per-category CSVs into a single DataFrame.
  SplitDataLoader — loads already-preprocessed train/val/test splits.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class RawDataLoader:
    """
    Loads and concatenates all raw CSV files from the MERGED_CSV directory.

    Usage
    -----
    loader = RawDataLoader()
    df = loader.load(Path("extracted/MERGED_CSV"))
    """

    def load(self, data_dir: Path) -> pd.DataFrame:
        """
        Read every *.csv in *data_dir*, concatenate, and return.

        Parameters
        ----------
        data_dir : Path
            Directory containing the raw CICIoT2023 per-category CSV files.

        Returns
        -------
        pd.DataFrame
            Combined dataframe with original columns (Label included).
        """
        csv_files = sorted(data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")

        logger.info("Found %d CSV files in %s", len(csv_files), data_dir)

        frames = []
        for path in csv_files:
            logger.debug("Reading %s", path.name)
            frames.append(pd.read_csv(path, low_memory=False))

        df = pd.concat(frames, ignore_index=True)
        logger.info("Loaded raw data — shape: %s", df.shape)
        return df

    def load_clean(self, path: Path) -> pd.DataFrame:
        """
        Load the already-deduplicated, inf-fixed clean CSV.

        Parameters
        ----------
        path : Path
            Path to ciciot_clean.csv produced by the EDA step.
        """
        logger.info("Loading clean CSV from %s", path)
        df = pd.read_csv(path)
        logger.info("Loaded clean data — shape: %s", df.shape)
        return df


class SplitDataLoader:
    """
    Loads pre-processed, scaled train/val/test splits from disk.

    Usage
    -----
    loader = SplitDataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load_binary(
        Path("extracted")
    )
    """

    def load_binary(
        self, data_dir: Path
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
               pd.Series, pd.Series, pd.Series]:
        """Return (X_train, X_val, X_test, y_train, y_val, y_test) for binary task."""
        return self._load_split(data_dir, suffix="binary")

    def load_cat8(
        self, data_dir: Path
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
               pd.Series, pd.Series, pd.Series]:
        """Return (X_train, X_val, X_test, y_train, y_val, y_test) for 8-category task."""
        return self._load_split(data_dir, suffix="cat8")

    def load_balanced_cat8(
        self, data_dir: Path
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
               pd.Series, pd.Series, pd.Series]:
        """
        Load splits where X_train is the SMOTE-ENN balanced version.
        Val and test remain the original imbalanced splits.
        """
        logger.info("Loading balanced cat8 training split from %s", data_dir)

        X_train = pd.read_csv(data_dir / "X_train_balanced_cat8.csv")
        y_train = pd.read_csv(data_dir / "y_train_balanced_cat8.csv").squeeze()

        X_val = pd.read_csv(data_dir / "X_val_cat8.csv")
        y_val = pd.read_csv(data_dir / "y_val_cat8.csv").squeeze()

        X_test = pd.read_csv(data_dir / "X_test_cat8.csv")
        y_test = pd.read_csv(data_dir / "y_test_cat8.csv").squeeze()

        logger.info(
            "Balanced cat8 — train: %s (balanced), val: %s, test: %s",
            X_train.shape, X_val.shape, X_test.shape,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _load_split(
        self, data_dir: Path, suffix: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
               pd.Series, pd.Series, pd.Series]:
        logger.info("Loading %s split from %s", suffix, data_dir)

        X_train = pd.read_csv(data_dir / f"X_train_{suffix}.csv")
        X_val   = pd.read_csv(data_dir / f"X_val_{suffix}.csv")
        X_test  = pd.read_csv(data_dir / f"X_test_{suffix}.csv")

        y_train = pd.read_csv(data_dir / f"y_train_{suffix}.csv").squeeze()
        y_val   = pd.read_csv(data_dir / f"y_val_{suffix}.csv").squeeze()
        y_test  = pd.read_csv(data_dir / f"y_test_{suffix}.csv").squeeze()

        logger.info(
            "%s split — train: %s, val: %s, test: %s",
            suffix, X_train.shape, X_val.shape, X_test.shape,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
