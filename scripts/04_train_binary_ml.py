"""
Script 04 — Binary ML Baseline (Random Forest + XGBoost)

Reproduces notebook 03_Baseline_ML.ipynb:
  - Loads binary train/val/test splits
  - Trains RF and XGBoost
  - Evaluates on the imbalanced test set
  - Saves models, results CSV, and plots

Usage
-----
    python scripts/04_train_binary_ml.py

    python scripts/04_train_binary_ml.py \
        --data-dir     /data/extracted \
        --models-dir   /data/models/binary \
        --results-dir  /data/results/binary \
        --no-gpu           # skip XGBoost GPU
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
from ciciot_ids.models.ml_models import RandomForestModel, XGBoostModel
from ciciot_ids.training.trainer import MLTrainer
from ciciot_ids.visualization.plots import IDSPlotter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("04_train_binary_ml")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Binary ML Baseline — RF + XGBoost")
    parser.add_argument("--data-dir",    type=Path, default=root / "extracted")
    parser.add_argument("--models-dir",  type=Path, default=root / "models" / "binary")
    parser.add_argument("--results-dir", type=Path, default=root / "results" / "binary")
    parser.add_argument("--show-plots",  action="store_true")
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable XGBoost GPU acceleration",
    )
    parser.add_argument("--rf-estimators",  type=int, default=100)
    parser.add_argument("--xgb-estimators", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    plotter  = IDSPlotter(show=args.show_plots)
    metrics  = BinaryMetricsComputer()
    trainer  = MLTrainer(eval_set_on_fit=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Loading binary splits from %s …", args.data_dir)
    loader = SplitDataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load_binary(args.data_dir)

    logger.info(
        "Train: %s | Val: %s | Test: %s",
        X_train.shape, X_val.shape, X_test.shape,
    )

    # ── 2. Random Forest ──────────────────────────────────────────────────────
    logger.info("=== Random Forest ===")
    rf = RandomForestModel(
        n_estimators=args.rf_estimators,
        class_weight="balanced",
        verbose=1,
    )
    rf, rf_time = trainer.train(rf, X_train.values, y_train.values, X_val.values, y_val.values)
    rf.save(args.models_dir / "rf_model.pkl")

    rf_pred  = rf.predict(X_test.values)
    rf_prob  = rf.predict_proba(X_test.values)[:, 1]
    rf_m     = metrics.compute(y_test.values, rf_pred, rf_prob)

    plotter.plot_confusion_matrix_binary(
        y_test.values, rf_pred,
        title="Random Forest — Confusion Matrix (Test)",
        save_path=args.results_dir / "rf_confusion_matrix.png",
    )
    plotter.plot_feature_importance(
        rf.feature_importances_,
        X_train.columns.tolist(),
        "Random Forest",
        save_path=args.results_dir / "rf_feature_importance.png",
    )

    # ── 3. XGBoost ────────────────────────────────────────────────────────────
    logger.info("=== XGBoost ===")
    xgb_params: dict = dict(n_estimators=args.xgb_estimators)
    if not args.no_gpu:
        xgb_params.update(tree_method="hist", device="cuda")

    scale_pos = int(y_train.value_counts()[0]) / int(y_train.value_counts()[1])
    xgb_params["scale_pos_weight"] = scale_pos

    xgb = XGBoostModel(task="binary", **xgb_params)
    xgb, xgb_time = trainer.train(xgb, X_train.values, y_train.values, X_val.values, y_val.values)
    xgb.save(args.models_dir / "xgb_model.pkl")

    xgb_pred = xgb.predict(X_test.values)
    xgb_prob = xgb.predict_proba(X_test.values)[:, 1]
    xgb_m    = metrics.compute(y_test.values, xgb_pred, xgb_prob)

    plotter.plot_confusion_matrix_binary(
        y_test.values, xgb_pred,
        title="XGBoost — Confusion Matrix (Test)",
        save_path=args.results_dir / "xgb_confusion_matrix.png",
    )
    plotter.plot_feature_importance(
        xgb.feature_importances_,
        X_train.columns.tolist(),
        "XGBoost",
        save_path=args.results_dir / "xgb_feature_importance.png",
    )

    # ── 4. Results table ──────────────────────────────────────────────────────
    rows = [
        metrics.to_row(rf_m,  "Random Forest", rf_time),
        metrics.to_row(xgb_m, "XGBoost",       xgb_time),
    ]
    results_df = pd.DataFrame(rows)
    csv_path = args.results_dir / "ml_baseline_results_binary.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("\n%s", results_df.to_string(index=False))

    plotter.plot_metric_comparison(
        results_df,
        metrics=["F1 Macro", "ROC-AUC", "PR-AUC", "MCC"],
        title="Binary ML Baseline — CICIoT2023",
        save_path=args.results_dir / "ml_baseline_comparison_binary.png",
    )

    # ── 5. Experiment log ─────────────────────────────────────────────────────
    log = {
        "script": "04_train_binary_ml",
        "dataset": "CICIoT2023",
        "task": "binary",
        "rf":  {k: round(v, 4) if isinstance(v, float) else v
                for k, v in rf_m.items() if not isinstance(v, int)},
        "xgb": {k: round(v, 4) if isinstance(v, float) else v
                for k, v in xgb_m.items() if not isinstance(v, int)},
        "rf_train_time_s":  round(rf_time, 1),
        "xgb_train_time_s": round(xgb_time, 1),
    }
    (args.results_dir / "experiment_log_binary_ml.json").write_text(
        json.dumps(log, indent=2)
    )
    logger.info("Binary ML baseline complete.")


if __name__ == "__main__":
    main()
