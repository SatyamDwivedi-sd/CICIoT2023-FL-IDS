"""
Evaluation metrics (Single Responsibility: compute metrics only, no I/O).

Two computers, one per task type (Interface Segregation — binary and
multiclass metrics are separate concerns with different signatures):

  BinaryMetricsComputer     — full suite for binary classification.
  MulticlassMetricsComputer — per-class + aggregate for N-class problems.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class BaseMetricsComputer(ABC):
    """Abstract base — open for extension, closed for modification."""

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> dict:
        """Return a dict of metric_name → value."""


class BinaryMetricsComputer(BaseMetricsComputer):
    """
    Computes the full binary IDS metric suite used across notebooks 03/04.

    Includes: accuracy, F1-weighted, F1-macro, ROC-AUC, PR-AUC, MCC,
    per-class precision/recall, FPR, FNR, and raw confusion matrix counts.
    """

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
    ) -> dict:
        """
        Parameters
        ----------
        y_true : array-like of {0, 1}
        y_pred : array-like of {0, 1}
        y_prob : array-like of float, optional
            Predicted probability for the *positive* (attack) class.
            Required for ROC-AUC and PR-AUC.

        Returns
        -------
        dict with keys: accuracy, f1_weighted, f1_macro, roc_auc, pr_auc,
            mcc, precision_benign, recall_benign, precision_attack,
            recall_attack, fpr, fnr, tn, fp, fn, tp
        """
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            confusion_matrix,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics: dict = {
            "accuracy":         accuracy_score(y_true, y_pred),
            "f1_weighted":      f1_score(y_true, y_pred, average="weighted"),
            "f1_macro":         f1_score(y_true, y_pred, average="macro"),
            "mcc":              matthews_corrcoef(y_true, y_pred),
            "precision_benign": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
            "recall_benign":    recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            "precision_attack": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            "recall_attack":    recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            "fpr":              fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            "fnr":              fn / (fn + tp) if (fn + tp) > 0 else 0.0,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }

        if y_prob is not None:
            y_prob = np.asarray(y_prob)
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            metrics["pr_auc"]  = average_precision_score(y_true, y_prob)
        else:
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"]  = float("nan")

        logger.info(
            "Binary metrics — acc: %.4f  F1-mac: %.4f  FPR: %.4f  FNR: %.4f",
            metrics["accuracy"], metrics["f1_macro"],
            metrics["fpr"], metrics["fnr"],
        )
        return metrics

    def to_row(self, metrics: dict, model_name: str, train_time: float) -> dict:
        """Flatten metrics dict into a single results-table row."""
        return {
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "F1 Weighted": metrics["f1_weighted"],
            "F1 Macro": metrics["f1_macro"],
            "ROC-AUC": metrics.get("roc_auc", float("nan")),
            "PR-AUC":  metrics.get("pr_auc",  float("nan")),
            "MCC": metrics["mcc"],
            "Benign Precision": metrics["precision_benign"],
            "Benign Recall":    metrics["recall_benign"],
            "Attack Precision": metrics["precision_attack"],
            "Attack Recall":    metrics["recall_attack"],
            "FPR": metrics["fpr"],
            "FNR": metrics["fnr"],
            "Train Time (s)": train_time,
        }


class MulticlassMetricsComputer(BaseMetricsComputer):
    """
    Computes aggregate + per-class metrics for N-class problems.

    Used in notebooks 07/08.
    """

    def __init__(self, class_names: dict[int, str]) -> None:
        """
        Parameters
        ----------
        class_names : dict[int, str]
            Mapping from integer label to human-readable name.
            E.g. {0: 'DDoS', 1: 'DoS', ...}
        """
        self._class_names = class_names

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs,
    ) -> dict:
        """
        Returns
        -------
        dict with keys:
          accuracy, f1_macro, f1_weighted,
          per_class: {name: {precision, recall, f1, support}}
        """
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            f1_score,
        )

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        target_names = [
            self._class_names[i] for i in sorted(self._class_names)
        ]

        report = classification_report(
            y_true, y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )

        per_class = {
            name: {
                "precision": report[name]["precision"],
                "recall":    report[name]["recall"],
                "f1":        report[name]["f1-score"],
                "support":   int(report[name]["support"]),
            }
            for name in target_names
        }

        metrics = {
            "accuracy":    accuracy_score(y_true, y_pred),
            "f1_macro":    f1_score(y_true, y_pred, average="macro",    zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "per_class":   per_class,
        }

        logger.info(
            "Multiclass metrics — acc: %.4f  F1-mac: %.4f  F1-wt: %.4f",
            metrics["accuracy"], metrics["f1_macro"], metrics["f1_weighted"],
        )
        return metrics

    def print_report(
        self, metrics: dict, model_name: str, split_name: str
    ) -> None:
        """Pretty-print aggregate + per-class results."""
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"{model_name} — {split_name}")
        print(sep)
        print(f"  Accuracy:      {metrics['accuracy']:.4f}")
        print(f"  F1 (macro):    {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        print()
        header = f"  {'Category':<14} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Support':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, vals in metrics["per_class"].items():
            print(
                f"  {name:<14} {vals['precision']:>7.4f} "
                f"{vals['recall']:>7.4f} {vals['f1']:>7.4f} "
                f"{vals['support']:>10,}"
            )
