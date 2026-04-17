"""
Data preprocessing (Single Responsibility: transforms only — no I/O beyond saving artifacts).

Responsibilities:
  - Encode labels (binary, 8-category, fine-34)
  - Stratified train/val/test split
  - MinMaxScaler fit on train, apply to val/test
  - Persist all artifacts to disk
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from ciciot_ids.constants import CATEGORY_ENCODING, CATEGORY_MAP

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Turns a clean DataFrame into scaled, labelled train/val/test splits.

    All state (scalers, encoders) lives on the instance so it can be
    serialised independently of the data.

    Parameters
    ----------
    test_size : float
        Fraction of total data held out for the test set.
    val_size : float
        Fraction of *non-test* data held out for validation.
        Default gives ~10 % of total data.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        test_size: float = 0.20,
        val_size: float = 0.125,
        random_state: int = 42,
    ) -> None:
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self._scaler_binary: MinMaxScaler | None = None
        self._scaler_cat8: MinMaxScaler | None = None
        self._fine_le: LabelEncoder | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach three label columns to *df* (returns a copy).

        Columns added
        -------------
        target_binary  : 0 = Benign, 1 = Attack
        target_cat8    : integer 0-7 per CATEGORY_ENCODING
        target_fine34  : integer 0-33 (LabelEncoder on raw Label column)
        """
        df = df.copy()

        # Binary
        df["target_binary"] = (df["Label"] != "BENIGN").astype(int)

        # 8-category
        df["Category"] = df["Label"].map(CATEGORY_MAP)
        unmapped = df.loc[df["Category"].isna(), "Label"].unique()
        if len(unmapped):
            raise ValueError(f"Labels not in CATEGORY_MAP: {unmapped.tolist()}")
        df["target_cat8"] = df["Category"].map(CATEGORY_ENCODING).astype(int)

        # Fine 34-class
        self._fine_le = LabelEncoder()
        df["target_fine34"] = self._fine_le.fit_transform(df["Label"])

        logger.info(
            "Targets built — binary dist: %s",
            df["target_binary"].value_counts().to_dict(),
        )
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return feature column names (excludes Label, Category, target_*)."""
        exclude = {"Label", "Category", "target_binary", "target_cat8", "target_fine34"}
        return [c for c in df.columns if c not in exclude]

    def split_and_scale_binary(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
               pd.Series, pd.Series, pd.Series]:
        """
        Stratified split + MinMaxScaler for the binary task.

        Returns
        -------
        X_train, X_val, X_test, y_train, y_val, y_test  (all scaled DataFrames / Series)
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self._split(X, y)
        self._scaler_binary = MinMaxScaler()
        X_train, X_val, X_test = self._scale(
            self._scaler_binary, X_train, X_val, X_test, X.columns.tolist()
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def split_and_scale_cat8(
        self, X: pd.DataFrame, y_cat8: pd.Series, y_fine34: pd.Series
    ) -> tuple:
        """
        Stratified split + MinMaxScaler for the 8-category task.

        Returns
        -------
        X_train, X_val, X_test,
        y_train_cat8, y_val_cat8, y_test_cat8,
        y_train_fine34, y_val_fine34, y_test_fine34
        """
        # Two-pass split keeping cat8 and fine34 aligned
        (X_temp, X_test,
         yc_temp, yc_test,
         yf_temp, yf_test) = train_test_split(
            X, y_cat8, y_fine34,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_cat8,
        )
        (X_train, X_val,
         yc_train, yc_val,
         yf_train, yf_val) = train_test_split(
            X_temp, yc_temp, yf_temp,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=yc_temp,
        )

        self._scaler_cat8 = MinMaxScaler()
        X_train, X_val, X_test = self._scale(
            self._scaler_cat8, X_train, X_val, X_test, X.columns.tolist()
        )

        logger.info(
            "Cat8 split — train: %s, val: %s, test: %s",
            X_train.shape, X_val.shape, X_test.shape,
        )
        return (X_train, X_val, X_test,
                yc_train, yc_val, yc_test,
                yf_train, yf_val, yf_test)

    def save_artifacts(
        self,
        output_dir: Path,
        *,
        X_train_b: pd.DataFrame,
        X_val_b: pd.DataFrame,
        X_test_b: pd.DataFrame,
        y_train_b: pd.Series,
        y_val_b: pd.Series,
        y_test_b: pd.Series,
        X_train_m: pd.DataFrame,
        X_val_m: pd.DataFrame,
        X_test_m: pd.DataFrame,
        y_train_cat8: pd.Series,
        y_val_cat8: pd.Series,
        y_test_cat8: pd.Series,
        y_train_fine34: pd.Series,
        y_val_fine34: pd.Series,
        y_test_fine34: pd.Series,
        feature_cols: list[str],
    ) -> None:
        """Persist all splits, scalers, encoders, and metadata to *output_dir*."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Feature splits ------------------------------------------------
        for name, arr in [
            ("X_train_binary", X_train_b), ("X_val_binary", X_val_b),
            ("X_test_binary", X_test_b),
            ("X_train_cat8", X_train_m), ("X_val_cat8", X_val_m),
            ("X_test_cat8", X_test_m),
        ]:
            arr.to_csv(output_dir / f"{name}.csv", index=False)

        # --- Label splits --------------------------------------------------
        for name, arr in [
            ("y_train_binary", y_train_b), ("y_val_binary", y_val_b),
            ("y_test_binary", y_test_b),
            ("y_train_cat8", y_train_cat8), ("y_val_cat8", y_val_cat8),
            ("y_test_cat8", y_test_cat8),
            ("y_train_fine34", y_train_fine34), ("y_val_fine34", y_val_fine34),
            ("y_test_fine34", y_test_fine34),
        ]:
            arr.to_csv(output_dir / f"{name}.csv", index=False)

        # --- Encoders / scalers --------------------------------------------
        (output_dir / "feature_columns.json").write_text(
            json.dumps(feature_cols, indent=2)
        )
        cat8_mapping = {k: int(v) for k, v in CATEGORY_ENCODING.items()}
        (output_dir / "cat8_mapping.json").write_text(
            json.dumps(cat8_mapping, indent=2)
        )
        if self._fine_le is not None:
            fine34_mapping = {
                label: int(idx)
                for idx, label in enumerate(self._fine_le.classes_)
            }
            (output_dir / "fine34_mapping.json").write_text(
                json.dumps(fine34_mapping, indent=2)
            )
            pickle.dump(
                self._fine_le,
                (output_dir / "label_encoder_fine34.pkl").open("wb"),
            )

        if self._scaler_binary is not None:
            pickle.dump(
                self._scaler_binary,
                (output_dir / "scaler_binary.pkl").open("wb"),
            )
        if self._scaler_cat8 is not None:
            pickle.dump(
                self._scaler_cat8,
                (output_dir / "scaler_cat8.pkl").open("wb"),
            )

        # --- Summary -------------------------------------------------------
        summary = {
            "binary_split": {
                "train_shape": list(X_train_b.shape),
                "val_shape": list(X_val_b.shape),
                "test_shape": list(X_test_b.shape),
                "train_distribution": y_train_b.value_counts().to_dict(),
                "val_distribution": y_val_b.value_counts().to_dict(),
                "test_distribution": y_test_b.value_counts().to_dict(),
            },
            "cat8_split": {
                "train_shape": list(X_train_m.shape),
                "val_shape": list(X_val_m.shape),
                "test_shape": list(X_test_m.shape),
                "train_distribution": y_train_cat8.value_counts().sort_index().to_dict(),
                "val_distribution": y_val_cat8.value_counts().sort_index().to_dict(),
                "test_distribution": y_test_cat8.value_counts().sort_index().to_dict(),
            },
        }
        (output_dir / "split_summary.json").write_text(
            json.dumps(summary, indent=2)
        )

        logger.info("All preprocessing artifacts saved to %s", output_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
               pd.Series, pd.Series, pd.Series]:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y_temp,
        )
        logger.info(
            "Split — train: %s, val: %s, test: %s",
            X_train.shape, X_val.shape, X_test.shape,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def _scale(
        scaler: MinMaxScaler,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        columns: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=columns)
        X_val   = pd.DataFrame(scaler.transform(X_val),       columns=columns)
        X_test  = pd.DataFrame(scaler.transform(X_test),      columns=columns)
        return X_train, X_val, X_test
