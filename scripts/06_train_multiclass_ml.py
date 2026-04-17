"""
Script 06 — Multiclass ML Baseline (RF + XGBoost, 8-category)

Reproduces notebook 07_Baseline_Multiclass_ML.ipynb:
  - Loads balanced cat8 training split + imbalanced val/test
  - Trains RF and XGBoost on balanced data
  - Evaluates on the IMBALANCED test set (this is the key metric)
  - Saves models, per-class results CSV, confusion matrices, feature importances

Usage
-----
    python scripts/06_train_multiclass_ml.py

    python scripts/06_train_multiclass_ml.py \
        --data-dir     /data/extracted \
        --models-dir   /data/models/cat8 \
        --results-dir  /data/results/cat8 \
        --no-gpu
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ciciot_ids.constants import CATEGORY_NAMES
from ciciot_ids.data.loader import SplitDataLoader
from ciciot_ids.evaluation.metrics import MulticlassMetricsComputer
from ciciot_ids.models.ml_models import RandomForestModel, XGBoostModel
from ciciot_ids.training.trainer import MLTrainer
from ciciot_ids.visualization.plots import IDSPlotter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("06_train_multiclass_ml")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Multiclass ML Baseline — RF + XGBoost")
    parser.add_argument("--data-dir",    type=Path, default=root / "extracted")
    parser.add_argument("--models-dir",  type=Path, default=root / "models" / "cat8")
    parser.add_argument("--results-dir", type=Path, default=root / "results" / "cat8")
    parser.add_argument("--show-plots",  action="store_true")
    parser.add_argument("--no-gpu",      action="store_true")
    parser.add_argument("--rf-estimators",  type=int, default=200)   # nb07 v2: 200
    parser.add_argument("--xgb-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth",  type=int, default=8)     # nb07 v2: 8
    return parser.parse_args()


def _build_results_csv(
    models: dict,          # {model_name: metrics_dict}
    y_test: np.ndarray,
    categories: list[str],
) -> pd.DataFrame:
    rows = []
    for cat in categories:
        row = {"category": cat}
        for name, m in models.items():
            row[f"{name.lower()}_precision"] = m["per_class"][cat]["precision"]
            row[f"{name.lower()}_recall"]    = m["per_class"][cat]["recall"]
            row[f"{name.lower()}_f1"]        = m["per_class"][cat]["f1"]
            row["support"]                   = m["per_class"][cat]["support"]
        rows.append(row)

    for label in ("ACCURACY", "F1_MACRO", "F1_WEIGHTED"):
        key = label.lower().replace("_", "_")
        row = {"category": label, "support": len(y_test)}
        for name, m in models.items():
            val = {
                "ACCURACY": m["accuracy"],
                "F1_MACRO": m["f1_macro"],
                "F1_WEIGHTED": m["f1_weighted"],
            }[label]
            row[f"{name.lower()}_precision"] = val
            row[f"{name.lower()}_recall"]    = val
            row[f"{name.lower()}_f1"]        = val
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    plotter  = IDSPlotter(show=args.show_plots)
    mc       = MulticlassMetricsComputer(CATEGORY_NAMES)
    trainer  = MLTrainer(eval_set_on_fit=False)
    categories = [CATEGORY_NAMES[i] for i in sorted(CATEGORY_NAMES)]

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Loading balanced cat8 training split …")
    loader = SplitDataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load_balanced_cat8(args.data_dir)

    logger.info(
        "Train (balanced): %s | Val: %s | Test (imbalanced): %s",
        X_train.shape, X_val.shape, X_test.shape,
    )

    all_metrics: dict[str, dict] = {}
    all_times:   dict[str, float] = {}

    for ModelClass, model_name, ckpt_name, extra_params in [
        (
            RandomForestModel,
            "RF",
            "rf_model_cat8.pkl",
            # nb07 v2: 200 estimators, no class_weight (data already SMOTE-balanced)
            {"n_estimators": args.rf_estimators, "class_weight": None, "verbose": 1},
        ),
        (
            XGBoostModel,
            "XGBoost",
            "xgb_model_cat8.pkl",
            # nb07 v2: max_depth=8, multi:softmax, CPU by default (no GPU arg)
            {
                "n_estimators": args.xgb_estimators,
                "max_depth": args.xgb_max_depth,
                "objective": "multi:softprob",
                **({"tree_method": "hist", "device": "cuda"} if not args.no_gpu else {}),
            },
        ),
    ]:
        logger.info("=== %s ===", model_name)

        if ModelClass is RandomForestModel:
            model = RandomForestModel(**extra_params)
        else:
            model = XGBoostModel(task="multiclass", num_classes=8, **extra_params)

        model, elapsed = trainer.train(model, X_train.values, y_train.values)
        model.save(args.models_dir / ckpt_name)

        # Evaluate on imbalanced test set
        test_pred = model.predict(X_test.values)
        m = mc.compute(y_test.values, test_pred)
        mc.print_report(m, model_name, "Test (Imbalanced)")

        all_metrics[model_name] = m
        all_times[model_name]   = elapsed

        plotter.plot_confusion_matrix_multi(
            y_test.values, test_pred,
            class_names=categories,
            title=f"{model_name} — Confusion Matrix (Test, Imbalanced)",
            save_path=args.results_dir / f"{model_name.lower()}_confusion_matrix_test.png",
        )
        plotter.plot_feature_importance(
            model.feature_importances_,
            X_train.columns.tolist(),
            model_name,
            save_path=args.results_dir / f"{model_name.lower()}_feature_importance.png",
        )

    # ── Results ───────────────────────────────────────────────────────────────
    plotter.plot_perclass_f1(
        categories,
        {name: [m["per_class"][c]["f1"] for c in categories]
         for name, m in all_metrics.items()},
        title="Per-Class F1 — RF vs XGBoost\nTrained balanced | Tested imbalanced",
        save_path=args.results_dir / "ml_baseline_f1_comparison.png",
    )

    results_df = _build_results_csv(all_metrics, y_test.values, categories)
    csv_path = args.results_dir / "results_ml_baseline_cat8.csv"
    results_df.to_csv(csv_path, index=False)

    log = {
        "script": "06_train_multiclass_ml",
        "dataset": "CICIoT2023",
        "task": "cat8_multiclass",
        "train_balanced": True,
        "test_imbalanced": True,
        "models": {
            name: {
                "accuracy":    round(m["accuracy"],    4),
                "f1_macro":    round(m["f1_macro"],    4),
                "f1_weighted": round(m["f1_weighted"], 4),
                "train_time_s": round(t, 1),
            }
            for (name, m), t in zip(all_metrics.items(), all_times.values())
        },
    }
    (args.results_dir / "experiment_log_multiclass_ml.json").write_text(
        json.dumps(log, indent=2)
    )

    logger.info("Multiclass ML baseline complete.")


if __name__ == "__main__":
    main()
