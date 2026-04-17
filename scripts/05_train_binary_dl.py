"""
Script 05 — Binary DL Baseline (MLP + 1D CNN, PyTorch)

Reproduces notebook 04_Baseline_DL.ipynb (converted to PyTorch):
  - Loads binary train/val/test splits
  - Trains MLP and 1D CNN with best-val-F1 checkpointing
  - Evaluates on the imbalanced test set
  - Saves models, results CSV, training curves, and plots

Usage
-----
    python scripts/05_train_binary_dl.py

    python scripts/05_train_binary_dl.py \
        --data-dir     /data/extracted \
        --models-dir   /data/models/binary \
        --results-dir  /data/results/binary \
        --epochs       20 \
        --batch-size   4096
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ciciot_ids.data.loader import SplitDataLoader
from ciciot_ids.evaluation.metrics import BinaryMetricsComputer
from ciciot_ids.models.dl_models import CNN1DModel, MLPModel
from ciciot_ids.training.trainer import DLTrainer
from ciciot_ids.visualization.plots import IDSPlotter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("05_train_binary_dl")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Binary DL Baseline — MLP + 1D CNN")
    parser.add_argument("--data-dir",    type=Path, default=root / "extracted")
    parser.add_argument("--models-dir",  type=Path, default=root / "models" / "binary")
    parser.add_argument("--results-dir", type=Path, default=root / "results" / "binary")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch-size",  type=int,   default=4096)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--show-plots",  action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    plotter = IDSPlotter(show=args.show_plots)
    mc      = BinaryMetricsComputer()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Loading binary splits …")
    loader = SplitDataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load_binary(args.data_dir)

    n_features = X_train.shape[1]
    X_tr, X_v, X_te = X_train.values, X_val.values, X_test.values
    y_tr, y_v, y_te = y_train.values, y_val.values, y_test.values

    logger.info(
        "Train: %s | Val: %s | Test: %s | Features: %d",
        X_train.shape, X_val.shape, X_test.shape, n_features,
    )

    trainer_cfg = dict(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        task="binary",
    )

    results_rows = []
    histories    = {}

    for ModelClass, model_name, ckpt_name in [
        (MLPModel,   "MLP",    "mlp_best.pt"),
        (CNN1DModel, "1D CNN", "cnn_best.pt"),
    ]:
        logger.info("=== %s ===", model_name)

        model = ModelClass(
            input_dim=n_features,
            num_classes=1,    # binary → single logit
            dropout=args.dropout,
        )
        logger.info("%s parameters: %d", model_name, model.num_params)

        trainer = DLTrainer(
            **trainer_cfg,
            checkpoint_path=args.models_dir / ckpt_name,
        )
        model, history, elapsed = trainer.train(model, X_tr, y_tr, X_v, y_v)
        histories[model_name] = history

        # ── Evaluate ──────────────────────────────────────────────────────
        pred  = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]
        m     = mc.compute(y_te, pred, proba)

        logger.info(
            "%s — acc: %.4f  F1-mac: %.4f  FPR: %.4f  FNR: %.4f",
            model_name, m["accuracy"], m["f1_macro"], m["fpr"], m["fnr"],
        )

        plotter.plot_confusion_matrix_binary(
            y_te, pred,
            title=f"{model_name} — Confusion Matrix (Test)",
            save_path=args.results_dir / f"{'mlp' if model_name == 'MLP' else 'cnn'}_confusion_matrix.png",
        )
        model.save(args.models_dir / f"{'mlp' if model_name == 'MLP' else 'cnn'}_model.pt")

        row = mc.to_row(m, model_name, elapsed)
        row["Type"] = "DL"
        results_rows.append(row)

    # ── Results table ─────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results_rows)
    csv_path = args.results_dir / "dl_baseline_results_binary.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("\n%s", results_df.to_string(index=False))

    plotter.plot_side_by_side_confusion(
        y_te,
        {r["Model"]: model.predict(X_te) for r in results_rows},
        save_path=args.results_dir / "dl_confusion_matrices_binary.png",
    )
    plotter.plot_metric_comparison(
        results_df,
        metrics=["F1 Macro", "ROC-AUC", "PR-AUC", "MCC"],
        title="Binary DL Baseline — CICIoT2023",
        save_path=args.results_dir / "dl_baseline_comparison_binary.png",
    )
    plotter.plot_training_curves(
        histories,
        save_path=args.results_dir / "dl_training_curves_binary.png",
    )

    # ── Training history ──────────────────────────────────────────────────────
    history_path = args.results_dir / "dl_training_history_binary.json"
    history_path.write_text(json.dumps(histories, indent=2))

    logger.info("Binary DL baseline complete.")


if __name__ == "__main__":
    main()
