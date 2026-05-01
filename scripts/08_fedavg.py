"""Script 08 — FedAvg simulation (MLP, CICIoT2023 cat8, Stage 2).

Mirrors experiments/stage2_fl/notebooks/10_FL_FedAvg.ipynb exactly.
Centralised baseline (scripts/07) must already exist before running this.

Outputs (all suffixed with partition mode so runs do not overwrite):
  fedavg/fedavg_round_log_{mode}.json
  results/fedavg_final_metrics_{mode}.json
  results/fedavg_final_metrics_{mode}.csv
  plots/fedavg_curves_{mode}.png
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')          # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── Fixed hyperparameters (match Stage 1 DL baseline) ────────────────────────
DROPOUT = 0.3
LR      = 1e-4

PARTITION_FILES = {
    'iid':          'iid_client_indices.npz',
    'dirichlet_03': 'dirichlet_03_client_indices.npz',
    'dirichlet_01': 'dirichlet_01_client_indices.npz',
}

# ── Project root: scripts/ → project root ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='FedAvg simulation — CICIoT2023 cat8')
    p.add_argument('--partition-mode',
                   choices=['iid', 'dirichlet_03', 'dirichlet_01'],
                   default='iid')
    p.add_argument('--rounds',        type=int,  default=20)
    p.add_argument('--local-epochs',  type=int,  default=3)
    p.add_argument('--batch-size',    type=int,  default=4096)
    p.add_argument('--seed',          type=int,  default=42)
    p.add_argument('--data-dir',      type=Path, default=ROOT / 'extracted')
    p.add_argument('--partition-dir', type=Path,
                   default=ROOT / 'experiments' / 'stage2_fl' / 'partitioning')
    p.add_argument('--fedavg-dir',    type=Path,
                   default=ROOT / 'experiments' / 'stage2_fl' / 'fedavg')
    p.add_argument('--results-dir',   type=Path,
                   default=ROOT / 'experiments' / 'stage2_fl' / 'results')
    p.add_argument('--plots-dir',     type=Path,
                   default=ROOT / 'experiments' / 'stage2_fl' / 'plots')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model — identical to Stage 1 multiclass DL baseline (scripts/07)
# ─────────────────────────────────────────────────────────────────────────────

def build_mlp(n_feat, n_cls, dropout=DROPOUT, lr=LR):
    """MLP: 256→128→64→n_cls with BatchNorm + Dropout — identical to Stage 1."""
    model = keras.Sequential([
        layers.Input(shape=(n_feat,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout * 0.67),
        layers.Dense(n_cls, activation='softmax'),
    ], name='MLP_CAT8')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# FedAvg core — mirrors nb10 exactly
# ─────────────────────────────────────────────────────────────────────────────

def average_weights(weight_list, sizes):
    """Standard FedAvg aggregation: w_global = sum_c (n_c / N) * w_c."""
    total = sum(sizes)
    avg = []
    for layer_arrays in zip(*weight_list):
        weighted_sum = sum(w * (s / total) for w, s in zip(layer_arrays, sizes))
        avg.append(weighted_sum)
    return avg


def train_client(global_weights, X_c, y_c, n_feat, n_cls, args):
    """Initialise a fresh model from global_weights, train local_epochs, return weights.

    A new model is instantiated per call to avoid Keras state accumulation
    across clients — identical behaviour to the notebook.
    """
    model = build_mlp(n_feat, n_cls)
    model.set_weights(global_weights)
    model.fit(
        X_c, y_c,
        epochs=args.local_epochs,
        batch_size=args.batch_size,
        verbose=0,
    )
    return model.get_weights(), len(X_c)


def evaluate_global(model, X, y, batch_size):
    """Predict on X, return accuracy / f1_macro / f1_weighted."""
    y_pred = np.argmax(model.predict(X, batch_size=batch_size, verbose=0), axis=1)
    return {
        'accuracy':    float(accuracy_score(y, y_pred)),
        'f1_macro':    float(f1_score(y, y_pred, average='macro',    zero_division=0)),
        'f1_weighted': float(f1_score(y, y_pred, average='weighted', zero_division=0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — fully headless (Agg backend set at module level)
# ─────────────────────────────────────────────────────────────────────────────

def plot_and_save_curves(round_log, args, num_clients, plots_dir):
    rounds  = [e['round']       for e in round_log]
    val_acc = [e['accuracy']    for e in round_log]
    val_f1m = [e['f1_macro']    for e in round_log]
    val_f1w = [e['f1_weighted'] for e in round_log]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(rounds, val_acc, marker='o', linewidth=1.5, markersize=4,
                 color='steelblue')
    axes[0].set_title(f'Val Accuracy over Rounds\n({args.partition_mode})')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Accuracy')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rounds, val_f1m, marker='o', linewidth=1.5, markersize=4,
                 label='F1 Macro', color='steelblue')
    axes[1].plot(rounds, val_f1w, marker='s', linewidth=1.5, markersize=4,
                 label='F1 Weighted', color='darkorange')
    axes[1].set_title(f'Val F1 over Rounds\n({args.partition_mode})')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('F1 Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f'FedAvg — {args.partition_mode}  '
        f'({num_clients} clients, {args.rounds} rounds, {args.local_epochs} local epochs)',
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    plot_path = plots_dir / f'fedavg_curves_{args.partition_mode}.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Plot         → {plot_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    for d in [args.fedavg_dir, args.results_dir, args.plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print('=== FedAvg — CICIoT2023 cat8 ===')
    print(f'partition_mode : {args.partition_mode}')
    print(f'rounds         : {args.rounds}')
    print(f'local_epochs   : {args.local_epochs}')
    print(f'batch_size     : {args.batch_size}')
    print(f'seed           : {args.seed}')
    print(f'data_dir       : {args.data_dir}')
    print(f'partition_dir  : {args.partition_dir}')
    print(f'fedavg_dir     : {args.fedavg_dir}')
    print(f'results_dir    : {args.results_dir}')
    print(f'plots_dir      : {args.plots_dir}')
    print()

    # ── Load data ─────────────────────────────────────────────────────────────
    print('Loading data...')
    t0 = time.time()

    X_train = pd.read_csv(args.data_dir / 'X_train_cat8.csv').to_numpy(dtype=np.float32)
    y_train = pd.read_csv(args.data_dir / 'y_train_cat8.csv').squeeze().to_numpy(dtype=np.int32)
    X_val   = pd.read_csv(args.data_dir / 'X_val_cat8.csv').to_numpy(dtype=np.float32)
    y_val   = pd.read_csv(args.data_dir / 'y_val_cat8.csv').squeeze().to_numpy(dtype=np.int32)
    X_test  = pd.read_csv(args.data_dir / 'X_test_cat8.csv').to_numpy(dtype=np.float32)
    y_test  = pd.read_csv(args.data_dir / 'y_test_cat8.csv').squeeze().to_numpy(dtype=np.int32)

    with open(args.data_dir / 'cat8_mapping.json') as f:
        cat8_mapping = json.load(f)
    id_to_category = {int(v): k for k, v in cat8_mapping.items()}
    num_classes    = len(cat8_mapping)
    class_names    = [id_to_category[i] for i in range(num_classes)]
    n_features     = X_train.shape[1]

    print(f'Loaded in {time.time() - t0:.1f}s')
    print(f'X_train: {X_train.shape}  y_train: {y_train.shape}')
    print(f'X_val:   {X_val.shape}    y_val:   {y_val.shape}')
    print(f'X_test:  {X_test.shape}   y_test:  {y_test.shape}')
    print(f'n_features={n_features}  num_classes={num_classes}')
    print(f'class_names: {class_names}')
    print()

    # ── Load partition ────────────────────────────────────────────────────────
    npz_path = args.partition_dir / PARTITION_FILES[args.partition_mode]
    npz = np.load(npz_path)
    num_clients    = len([k for k in npz.files if k.startswith('client_')])
    client_indices = [npz[f'client_{c}'] for c in range(num_clients)]
    npz.close()

    sizes = [len(idx) for idx in client_indices]
    print(f'Partition : {args.partition_mode}  ({npz_path.name})')
    print(f'Clients   : {num_clients}')
    print(f'Client sizes : {sizes}')
    print(f'Min: {min(sizes):,}  Max: {max(sizes):,}  Total: {sum(sizes):,}')
    print()

    # ── FedAvg loop ───────────────────────────────────────────────────────────
    print(f'Starting FedAvg | partition={args.partition_mode} | '
          f'rounds={args.rounds} | local_epochs={args.local_epochs}')
    print()

    global_model = build_mlp(n_features, num_classes)
    round_log    = []
    t_start      = time.time()

    for rnd in range(1, args.rounds + 1):
        t_rnd          = time.time()
        global_weights = global_model.get_weights()

        # Broadcast → local train
        client_weight_list = []
        client_size_list   = []
        for c in range(num_clients):
            idx  = client_indices[c]
            w, s = train_client(
                global_weights, X_train[idx], y_train[idx],
                n_features, num_classes, args,
            )
            client_weight_list.append(w)
            client_size_list.append(s)

        # Aggregate
        new_weights = average_weights(client_weight_list, client_size_list)
        global_model.set_weights(new_weights)

        # Validation evaluation
        val_m = evaluate_global(global_model, X_val, y_val, args.batch_size)
        entry = {'round': rnd, **val_m, 'elapsed_s': round(time.time() - t_rnd, 1)}
        round_log.append(entry)

        print(
            f'Round {rnd:2d}/{args.rounds} | '
            f'acc={val_m["accuracy"]:.4f} | '
            f'f1_macro={val_m["f1_macro"]:.4f} | '
            f'f1_weighted={val_m["f1_weighted"]:.4f} | '
            f'{entry["elapsed_s"]:.0f}s'
        )

    print(f'\nFedAvg complete — total time: {(time.time() - t_start) / 60:.1f} min')
    print()

    # ── Final test evaluation ─────────────────────────────────────────────────
    print('=== Final evaluation on TEST set ===')
    print()
    y_test_pred = np.argmax(
        global_model.predict(X_test, batch_size=args.batch_size, verbose=0), axis=1
    )
    test_acc = float(accuracy_score(y_test, y_test_pred))
    test_f1m = float(f1_score(y_test, y_test_pred, average='macro',    zero_division=0))
    test_f1w = float(f1_score(y_test, y_test_pred, average='weighted', zero_division=0))

    print(f'Test accuracy   : {test_acc:.4f}')
    print(f'Test F1 macro   : {test_f1m:.4f}')
    print(f'Test F1 weighted: {test_f1w:.4f}')
    print()
    print(classification_report(y_test, y_test_pred, target_names=class_names,
                                zero_division=0))

    per_class_f1  = f1_score(y_test, y_test_pred, average=None, zero_division=0)
    per_class_dict = {
        f'f1_{cls}': float(per_class_f1[i])
        for i, cls in enumerate(class_names)
    }

    # ── Save outputs ──────────────────────────────────────────────────────────
    mode = args.partition_mode

    # Config block included in every output
    config = {
        'partition_mode': mode,
        'rounds':         args.rounds,
        'local_epochs':   args.local_epochs,
        'batch_size':     args.batch_size,
        'seed':           args.seed,
    }

    # Round log JSON: metadata wrapper + rounds array
    log_path = args.fedavg_dir / f'fedavg_round_log_{mode}.json'
    with open(log_path, 'w') as f:
        json.dump({**config, 'round_log': round_log}, f, indent=2)
    print(f'Round log    → {log_path}')

    # Final metrics JSON
    final_metrics = {
        **config,
        'test_accuracy':    test_acc,
        'test_f1_macro':    test_f1m,
        'test_f1_weighted': test_f1w,
        **per_class_dict,          # f1_DDoS, f1_DoS, ...
    }
    metrics_json = args.results_dir / f'fedavg_final_metrics_{mode}.json'
    with open(metrics_json, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f'Metrics JSON → {metrics_json}')

    # Final metrics CSV: config rows, then overall metrics, then per-class F1
    rows = (
        [{'metric': k, 'value': v} for k, v in config.items()]
        + [{'metric': 'test_accuracy',    'value': test_acc},
           {'metric': 'test_f1_macro',    'value': test_f1m},
           {'metric': 'test_f1_weighted', 'value': test_f1w}]
        + [{'metric': k, 'value': v} for k, v in per_class_dict.items()]
    )
    metrics_csv = args.results_dir / f'fedavg_final_metrics_{mode}.csv'
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    print(f'Metrics CSV  → {metrics_csv}')

    # Plot
    plot_and_save_curves(round_log, args, num_clients, args.plots_dir)

    print()
    print('── All outputs saved ──')
    print(f'  fedavg/   fedavg_round_log_{mode}.json')
    print(f'  results/  fedavg_final_metrics_{mode}.json')
    print(f'  results/  fedavg_final_metrics_{mode}.csv')
    print(f'  plots/    fedavg_curves_{mode}.png')


if __name__ == '__main__':
    main()
