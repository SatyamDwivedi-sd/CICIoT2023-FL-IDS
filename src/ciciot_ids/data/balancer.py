"""
Class imbalance handling (Single Responsibility: balancing only).

Strategy used in notebook 06:
  1. RandomUnderSampler  — shrink majority classes to target counts.
  2. SMOTE-ENN           — oversample + clean minority classes.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ClassBalancer:
    """
    Two-phase balancer: undersampling of majority classes followed by
    SMOTE-ENN oversampling of minority classes.

    Parameters
    ----------
    under_targets : dict[int, int]
        {class_label: target_count} for classes to undersample.
    over_targets : dict[int, int]
        {class_label: target_count} for classes to oversample with SMOTE-ENN.
    random_state : int
        Seed for reproducibility.
    n_jobs : int
        Parallel jobs forwarded to SMOTE-ENN (-1 = all cores).
    """

    def __init__(
        self,
        under_targets: dict[int, int],
        over_targets: dict[int, int],
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.under_targets = under_targets
        self.over_targets = over_targets
        self.random_state = random_state
        self.n_jobs = n_jobs

    def balance(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Apply undersampling then SMOTE-ENN.

        Parameters
        ----------
        X : pd.DataFrame  — feature matrix (already scaled).
        y : pd.Series     — integer class labels.

        Returns
        -------
        X_bal, y_bal  — balanced feature matrix and labels.
        """
        try:
            from imblearn.combine import SMOTEENN
            from imblearn.under_sampling import RandomUnderSampler
        except ImportError as exc:
            raise ImportError(
                "imbalanced-learn is required for balancing. "
                "Install with: pip install imbalanced-learn"
            ) from exc

        logger.info("Original class counts: %s", dict(sorted(Counter(y).items())))

        # ── Phase 1: undersampling ──────────────────────────────────────────
        current = Counter(y)
        under_strategy = {
            k: v for k, v in self.under_targets.items() if current[k] > v
        }

        if under_strategy:
            logger.info("Applying RandomUnderSampler with targets: %s", under_strategy)
            rus = RandomUnderSampler(
                sampling_strategy=under_strategy,
                random_state=self.random_state,
            )
            X_np, y_np = rus.fit_resample(X.values, y.values)
            logger.info(
                "After undersampling — shape: %s, counts: %s",
                X_np.shape, dict(sorted(Counter(y_np).items())),
            )
        else:
            X_np, y_np = X.values, y.values
            logger.info("No undersampling required.")

        # ── Phase 2: SMOTE-ENN oversampling ────────────────────────────────
        if self.over_targets:
            logger.info("Applying SMOTE-ENN with targets: %s", self.over_targets)
            smote_enn = SMOTEENN(
                sampling_strategy=self.over_targets,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            X_np, y_np = smote_enn.fit_resample(X_np, y_np)
            logger.info(
                "After SMOTE-ENN — shape: %s, counts: %s",
                X_np.shape, dict(sorted(Counter(y_np).items())),
            )

        X_bal = pd.DataFrame(X_np, columns=X.columns)
        y_bal = pd.Series(y_np, name=y.name)
        return X_bal, y_bal

    def save_balanced(
        self,
        output_dir: Path,
        X_bal: pd.DataFrame,
        y_bal: pd.Series,
        prefix: str = "balanced_cat8",
    ) -> None:
        """Save balanced feature matrix and labels to CSV."""
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        X_bal.to_csv(output_dir / f"X_train_{prefix}.csv", index=False)
        y_bal.to_csv(output_dir / f"y_train_{prefix}.csv", index=False)

        summary = {
            "prefix": prefix,
            "final_shape": list(X_bal.shape),
            "final_counts": {
                str(k): int(v)
                for k, v in sorted(Counter(y_bal).items())
            },
            "under_targets": {str(k): v for k, v in self.under_targets.items()},
            "over_targets":  {str(k): v for k, v in self.over_targets.items()},
        }
        (output_dir / f"balancing_summary_{prefix}.json").write_text(
            json.dumps(summary, indent=2)
        )

        logger.info("Balanced data saved to %s with prefix '%s'", output_dir, prefix)
