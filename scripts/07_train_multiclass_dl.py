"""
Script 07 — Multiclass DL Baseline (MLP + 1D CNN, TensorFlow/Keras, 8-category)

Mirrors notebook 08_Baseline_Multiclass_DL.ipynb (v2 — Keras version):
  - Loads balanced cat8 training split + imbalanced val/test
  - Trains MLP and 1D CNN with EarlyStopping + ModelCheckpoint
  - Evaluates on the IMBALANCED test set
  - Saves models, per-class results CSV, confusion matrices, training curves

Usage
-----
Default run (matches notebook — no class weights):

    python scripts/07_train_multiclass_dl.py

    python scripts/07_train_multiclass_dl.py \
        --data-dir     /data/extracted \
        --models-dir   /data/models/cat8 \
        --results-dir  /data/results/cat8 \
        --epochs       15 \
        --batch-size   4096

Ablation run (inverse-frequency class weights applied to loss):

    python scripts/07_train_multiclass_dl.py \
        --data-dir     /data/extracted \
        --models-dir   /data/models/cat8 \
        --results-dir  /data/results/cat8 \
        --epochs       15 \
        --batch-size   4096 \
        --use-class-weights
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
from ciciot_ids.visualization.plots import IDSPlotter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("07_train_multiclass_dl")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Multiclass DL Baseline — MLP + 1D CNN (Keras)")
    parser.add_argument("--data-dir",    type=Path, default=root / "extracted")
    parser.add_argument("--models-dir",  type=Path, default=root / "models" / "cat8")
    parser.add_argument("--results-dir", type=Path, default=root / "results" / "cat8")
    parser.add_argument("--epochs",      type=int,   default=15)
    parser.add_argument("--batch-size",  type=int,   default=4096)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--patience",    type=int,   default=3)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--show-plots",  action="store_true")
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        default=False,
        help="Apply inverse-frequency class weights to loss (ablation only — default off matches notebook)",
    )
    return parser.parse_args()


class ValF1Callback:
    """Compute val F1-macro after every epoch and append to self.val_f1."""

    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, batch_size: int) -> None:
        from tensorflow import keras
        self._base = keras.callbacks.Callback()
        self._base.on_epoch_end = self._on_epoch_end
        self.X_val      = X_val
        self.y_val      = y_val
        self.batch_size = batch_size
        self.val_f1: list[float] = []

    def _on_epoch_end(self, epoch: int, logs=None) -> None:
        from sklearn.metrics import f1_score
        prob  = self._base.model.predict(self.X_val, batch_size=self.batch_size, verbose=0)
        pred  = np.argmax(prob, axis=1)
        score = f1_score(self.y_val, pred, average="macro", zero_division=0)
        self.val_f1.append(float(score))
        if logs is not None:
            logs["val_f1_macro"] = score

    @property
    def callback(self):
        return self._base


def build_class_weights(y_train: np.ndarray, num_classes: int) -> dict:
    """Compute inverse-frequency class weights: weight_i = total / (n_classes * count_i)."""
    from collections import Counter
    counts = Counter(y_train)
    total  = len(y_train)
    return {
        cls: total / (num_classes * counts[cls])
        for cls in range(num_classes)
    }


def build_mlp(n_features: int, num_classes: int, dropout: float, lr: float):
    """MLP: 256→128→64→num_classes with BatchNorm + Dropout."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout * 0.67),
        layers.Dense(num_classes, activation="softmax"),
    ], name="MLP_CAT8")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(n_features: int, num_classes: int, dropout: float, lr: float):
    """1D CNN: Conv(64)→MaxPool→Conv(128)→MaxPool→Dense(64)→num_classes."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(n_features, 1)),
        layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(dropout),
        layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(dropout),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout * 0.67),
        layers.Dense(num_classes, activation="softmax"),
    ], name="CNN_CAT8")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> dict:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        classification_report, f1_score, matthews_corrcoef,
    )

    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
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
    return {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_weighted":       f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro":          f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "mcc":               matthews_corrcoef(y_true, y_pred),
        "per_class":         per_class,
    }


def print_report(metrics: dict, model_name: str) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n{model_name} — Test (Imbalanced)\n{sep}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  F1 Macro:          {metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted:       {metrics['f1_weighted']:.4f}")
    print(f"  MCC:               {metrics['mcc']:.4f}")
    print()
    print(f"  {'Class':<14} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Support':>10}")
    print("  " + "-" * 46)
    for name, v in metrics["per_class"].items():
        print(f"  {name:<14} {v['precision']:>7.4f} {v['recall']:>7.4f} "
              f"{v['f1']:>7.4f} {v['support']:>10,}")


def _build_results_csv(
    models: dict, train_metrics: dict, y_test: np.ndarray, categories: list[str]
) -> pd.DataFrame:
    rows = []
    for cat in categories:
        row = {"category": cat}
        for name, m in models.items():
            prefix = name.lower().replace(" ", "_")
            row[f"{prefix}_precision"] = m["per_class"][cat]["precision"]
            row[f"{prefix}_recall"]    = m["per_class"][cat]["recall"]
            row[f"{prefix}_f1"]        = m["per_class"][cat]["f1"]
            row["support"]             = m["per_class"][cat]["support"]
        rows.append(row)

    test_key_map = {
        "TEST_ACCURACY":          "accuracy",
        "TEST_BALANCED_ACCURACY": "balanced_accuracy",
        "TEST_F1_MACRO":          "f1_macro",
        "TEST_F1_WEIGHTED":       "f1_weighted",
        "TEST_MCC":               "mcc",
    }
    for label, key in test_key_map.items():
        row = {"category": label, "support": len(y_test)}
        for name, m in models.items():
            prefix = name.lower().replace(" ", "_")
            val = m[key]
            row[f"{prefix}_precision"] = val
            row[f"{prefix}_recall"]    = val
            row[f"{prefix}_f1"]        = val
        rows.append(row)

    train_key_map = {
        "TRAIN_ACCURACY":    "accuracy",
        "TRAIN_F1_MACRO":    "f1_macro",
        "TRAIN_F1_WEIGHTED": "f1_weighted",
    }
    for label, key in train_key_map.items():
        row = {"category": label, "support": None}
        for name, tm in train_metrics.items():
            prefix = name.lower().replace(" ", "_")
            val = tm[key]
            row[f"{prefix}_precision"] = val
            row[f"{prefix}_recall"]    = val
            row[f"{prefix}_f1"]        = val
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    import tensorflow as tf

    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    tf.random.set_seed(42)
    np.random.seed(42)

    plotter    = IDSPlotter(show=args.show_plots)
    categories = [CATEGORY_NAMES[i] for i in sorted(CATEGORY_NAMES)]
    num_classes = 8

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Loading balanced cat8 splits …")
    loader = SplitDataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load_balanced_cat8(args.data_dir)

    n_features = X_train.shape[1]
    X_tr = X_train.values.astype(np.float32)
    X_v  = X_val.values.astype(np.float32)
    X_te = X_test.values.astype(np.float32)
    y_tr = y_train.values.astype(np.int32)
    y_v  = y_val.values.astype(np.int32)
    y_te = y_test.values.astype(np.int32)

    logger.info(
        "Train (balanced): %s | Val: %s | Test (imbalanced): %s | Features: %d",
        X_train.shape, X_val.shape, X_test.shape, n_features,
    )

    # ── 2. Class weights (ablation only — off by default) ─────────────────────
    class_weight = None
    if args.use_class_weights:
        class_weight = build_class_weights(y_tr, num_classes)
        logger.info("Class weights: %s", {CATEGORY_NAMES[k]: round(v, 3)
                                           for k, v in class_weight.items()})

    from tensorflow import keras

    all_metrics:       dict[str, dict] = {}
    all_train_metrics: dict[str, dict] = {}
    all_histories:     dict[str, dict] = {}
    all_times:         dict[str, float] = {}

    import time

    for model_name, model_fn, X_val_in, X_test_in in [
        ("MLP",   build_mlp, X_v,                X_te),
        ("1D CNN", build_cnn, X_v.reshape(-1, n_features, 1),
                              X_te.reshape(-1, n_features, 1)),
    ]:
        logger.info("=== %s ===", model_name)
        safe_name = model_name.lower().replace(" ", "_")

        model = model_fn(n_features, num_classes, args.dropout, args.lr)
        model.summary(print_fn=logger.info)

        X_train_in = X_tr if model_name == "MLP" else X_tr.reshape(-1, n_features, 1)

        val_f1_cb = ValF1Callback(X_val_in, y_v, args.batch_size)
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience,
                restore_best_weights=True, verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(args.models_dir / f"{safe_name}_cat8_best.keras"),
                monitor="val_loss", save_best_only=True, verbose=1,
            ),
            val_f1_cb.callback,
        ]

        t0 = time.time()
        history = model.fit(
            X_train_in, y_tr,
            validation_data=(X_val_in, y_v),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )
        elapsed = time.time() - t0

        # ── Evaluate on imbalanced test set ───────────────────────────────
        prob  = model.predict(X_test_in, batch_size=args.batch_size, verbose=0)
        pred  = np.argmax(prob, axis=1)
        m     = compute_metrics(y_te, pred, categories)
        print_report(m, model_name)

        # ── Evaluate on balanced training set ─────────────────────────────
        train_prob = model.predict(X_train_in, batch_size=args.batch_size, verbose=0)
        train_pred = np.argmax(train_prob, axis=1)
        train_m    = compute_metrics(y_tr, train_pred, categories)

        all_metrics[model_name]       = m
        all_train_metrics[model_name] = train_m
        all_histories[model_name] = {
            "train_loss": history.history["loss"],
            "val_loss":   history.history["val_loss"],
            "val_f1":     val_f1_cb.val_f1,
        }
        all_times[model_name] = elapsed

        model.save(str(args.models_dir / f"{safe_name}_cat8_final.keras"))

        plotter.plot_confusion_matrix_multi(
            y_te, pred,
            class_names=categories,
            title=f"{model_name} — Confusion Matrix (Test, Imbalanced)",
            save_path=args.results_dir / f"{safe_name}_confusion_matrix_test.png",
        )

    # ── Results ───────────────────────────────────────────────────────────────
    plotter.plot_training_curves(
        all_histories,
        save_path=args.results_dir / "dl_training_curves_cat8.png",
    )
    plotter.plot_perclass_f1(
        categories,
        {name: [m["per_class"][c]["f1"] for c in categories]
         for name, m in all_metrics.items()},
        title="Per-Class F1 — MLP vs 1D CNN\nTrained balanced | Tested imbalanced",
        save_path=args.results_dir / "dl_baseline_f1_comparison.png",
    )

    results_df = _build_results_csv(all_metrics, all_train_metrics, y_te, categories)
    results_df.to_csv(args.results_dir / "results_dl_baseline_cat8.csv", index=False)

    history_bundle = {
        name: {"loss": h["train_loss"], "val_loss": h["val_loss"], "val_f1": h["val_f1"]}
        for name, h in all_histories.items()
    }
    (args.results_dir / "dl_training_history_cat8.json").write_text(
        json.dumps(history_bundle, indent=2)
    )

    log = {
        "script": "07_train_multiclass_dl",
        "framework": "tensorflow/keras",
        "dataset": "CICIoT2023",
        "task": "cat8_multiclass",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "use_class_weights": args.use_class_weights,
        "train_balanced": True,
        "test_imbalanced": True,
        "models": {
            name: {
                "test_accuracy":          round(m["accuracy"],                          4),
                "test_balanced_accuracy": round(m["balanced_accuracy"],                 4),
                "test_f1_macro":          round(m["f1_macro"],                          4),
                "test_f1_weighted":       round(m["f1_weighted"],                       4),
                "test_mcc":               round(m["mcc"],                               4),
                "train_accuracy":         round(all_train_metrics[name]["accuracy"],    4),
                "train_f1_macro":         round(all_train_metrics[name]["f1_macro"],    4),
                "train_f1_weighted":      round(all_train_metrics[name]["f1_weighted"], 4),
                "train_time_s":           round(all_times[name],                        1),
            }
            for name, m in all_metrics.items()
        },
    }
    (args.results_dir / "experiment_log_multiclass_dl.json").write_text(
        json.dumps(log, indent=2)
    )
    logger.info("Multiclass DL baseline complete.")


if __name__ == "__main__":
    main()
