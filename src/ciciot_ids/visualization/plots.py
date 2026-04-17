"""
Visualization utilities (Single Responsibility: only produce plots).

All plot methods save to disk when *save_path* is provided and optionally
display interactively when *show=True*. In non-interactive environments
(cluster, CI) set show=False.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _use_agg_if_headless() -> None:
    """Switch to non-interactive backend if no display is available."""
    import os
    if not os.environ.get("DISPLAY") and matplotlib.get_backend() != "Agg":
        matplotlib.use("Agg")


class IDSPlotter:
    """
    All project-specific plots in one place.

    Parameters
    ----------
    show : bool
        Display plots interactively (set False on headless servers).
    dpi : int
        Resolution for saved figures.
    """

    def __init__(self, show: bool = False, dpi: int = 150) -> None:
        _use_agg_if_headless()
        self.show = show
        self.dpi = dpi

    # ------------------------------------------------------------------
    # Class distribution
    # ------------------------------------------------------------------

    def plot_class_distribution(
        self,
        counts: pd.Series,
        title: str = "Class Distribution",
        save_path: Path | None = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(counts.index, counts.values, color="steelblue", edgecolor="white")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts.values) * 0.01,
                f"{val:,}", ha="center", fontsize=8,
            )
        plt.tight_layout()
        self._save_and_show(fig, save_path)

    def plot_balancing_comparison(
        self,
        before: pd.Series,
        after: dict[str, int],
        save_path: Path | None = None,
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ordered = list(after.keys())
        before_vals = before.reindex(ordered).values

        colors = ["red" if v < 100_000 else "orange" if v < 1_000_000 else "steelblue"
                  for v in before_vals]
        axes[0].bar(ordered, before_vals, color=colors)
        axes[0].set_title("BEFORE Balancing", fontsize=12, fontweight="bold")
        axes[0].tick_params(axis="x", rotation=30)

        after_vals = list(after.values())
        colors_after = ["steelblue"] * (len(after_vals) - 2) + ["coral"] * 2
        axes[1].bar(ordered, after_vals, color=colors_after)
        axes[1].set_title(
            "AFTER Balancing\nBlue=Undersample  Coral=SMOTE-ENN",
            fontsize=12, fontweight="bold",
        )
        axes[1].tick_params(axis="x", rotation=30)

        for ax, vals in [(axes[0], before_vals), (axes[1], after_vals)]:
            for i, v in enumerate(vals):
                ax.text(i, v + max(vals) * 0.01, f"{v:,}", ha="center", fontsize=8)

        plt.suptitle(
            "CICIoT2023 — Class Balancing Strategy",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        self._save_and_show(fig, save_path)

    # ------------------------------------------------------------------
    # Confusion matrices
    # ------------------------------------------------------------------

    def plot_confusion_matrix_binary(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix",
        save_path: Path | None = None,
    ) -> None:
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["Benign", "Attack"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(title, fontweight="bold")
        plt.tight_layout()
        self._save_and_show(fig, save_path)

    def plot_confusion_matrix_multi(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: list[str],
        title: str = "Confusion Matrix (Normalized)",
        save_path: Path | None = None,
    ) -> None:
        try:
            import seaborn as sns
        except ImportError:
            sns = None

        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred, normalize="true")
        fig, ax = plt.subplots(figsize=(10, 8))

        if sns is not None:
            sns.heatmap(
                cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, ax=ax, linewidths=0.5,
            )
        else:
            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticklabels(class_names)
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8)

        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title(title, fontsize=13)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        self._save_and_show(fig, save_path)

    def plot_side_by_side_confusion(
        self,
        y_true: np.ndarray,
        preds: dict[str, np.ndarray],
        save_path: Path | None = None,
    ) -> None:
        """Plot multiple confusion matrices side-by-side (for binary task)."""
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        n = len(preds)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, (name, pred) in zip(axes, preds.items()):
            cm = confusion_matrix(y_true, pred)
            ConfusionMatrixDisplay(cm, display_labels=["Benign", "Attack"]).plot(
                ax=ax, colorbar=False, cmap="Blues"
            )
            ax.set_title(f"{name} — Confusion Matrix", fontweight="bold")

        plt.tight_layout()
        self._save_and_show(fig, save_path)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def plot_feature_importance(
        self,
        importances: np.ndarray,
        feature_names: list[str],
        model_name: str,
        top_n: int = 20,
        save_path: Path | None = None,
    ) -> None:
        indices = np.argsort(importances)[::-1][:top_n]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            range(top_n),
            importances[indices][::-1],
            color="steelblue", edgecolor="white",
        )
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=9)
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(f"{model_name} — Top {top_n} Feature Importances", fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        self._save_and_show(fig, save_path)

    # ------------------------------------------------------------------
    # Comparison bars
    # ------------------------------------------------------------------

    def plot_metric_comparison(
        self,
        results_df: pd.DataFrame,
        metrics: list[str],
        title: str = "Model Comparison",
        save_path: Path | None = None,
    ) -> None:
        """Bar chart comparing multiple metrics across models."""
        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        models = results_df["Model"].tolist()
        for ax, metric in zip(axes, metrics):
            vals = results_df[metric].tolist()
            bars = ax.bar(models, vals, width=0.5)
            ax.set_title(metric, fontsize=12, fontweight="bold")
            ax.set_ylim(0, 1.0)
            ax.tick_params(axis="x", rotation=15)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", fontsize=9, fontweight="bold",
                )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save_and_show(fig, save_path)

    def plot_perclass_f1(
        self,
        categories: list[str],
        scores: dict[str, list[float]],
        title: str = "Per-Class F1",
        save_path: Path | None = None,
    ) -> None:
        """Grouped bar chart of per-class F1 for multiple models."""
        n_models = len(scores)
        x = np.arange(len(categories))
        width = 0.8 / n_models
        colors = ["steelblue", "coral", "seagreen", "orchid"]

        fig, ax = plt.subplots(figsize=(13, 6))
        for i, (model_name, f1s) in enumerate(scores.items()):
            offset = (i - n_models / 2 + 0.5) * width
            bars = ax.bar(x + offset, f1s, width, label=model_name,
                          color=colors[i % len(colors)], alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=30, ha="right")
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("F1-Score", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.legend(fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        self._save_and_show(fig, save_path)

    # ------------------------------------------------------------------
    # Training curves (DL)
    # ------------------------------------------------------------------

    def plot_training_curves(
        self,
        histories: dict[str, dict[str, list[float]]],
        save_path: Path | None = None,
    ) -> None:
        """
        Plot loss and val-F1 curves for multiple models.

        Parameters
        ----------
        histories : {model_name: {'train_loss': [...], 'val_f1': [...]}}
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = ["steelblue", "coral", "seagreen", "orchid"]

        for i, (name, hist) in enumerate(histories.items()):
            c = colors[i % len(colors)]
            axes[0].plot(hist["train_loss"], label=name, color=c)
            axes[1].plot(hist["val_f1"], label=name, color=c)

        for ax, ylabel, title in [
            (axes[0], "Loss", "Training Loss"),
            (axes[1], "F1 Macro", "Validation F1 Macro"),
        ]:
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight="bold")
            ax.legend()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.suptitle("DL Training Curves", fontsize=13)
        plt.tight_layout()
        self._save_and_show(fig, save_path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save_and_show(self, fig: plt.Figure, save_path: Path | None) -> None:
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info("Plot saved → %s", save_path)
        if self.show:
            plt.show()
        plt.close(fig)
