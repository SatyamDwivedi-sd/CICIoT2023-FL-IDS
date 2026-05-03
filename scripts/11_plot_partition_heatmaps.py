"""Create partition client/class count tables and heatmaps for Stage 2.

Reads:
  experiments/stage2_fl/partitioning/{mode}_client_indices.npz
  {data-dir}/y_train_cat8.csv
  {data-dir}/cat8_mapping.json

Writes:
  experiments/stage2_fl/results/partition_client_class_counts.csv
  experiments/stage2_fl/plots/partition_heatmap_iid.png
  experiments/stage2_fl/plots/partition_heatmap_dirichlet_03.png
  experiments/stage2_fl/plots/partition_heatmap_dirichlet_01.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PARTITION_FILES = {
    "iid": "iid_client_indices.npz",
    "dirichlet_03": "dirichlet_03_client_indices.npz",
    "dirichlet_01": "dirichlet_01_client_indices.npz",
}
MODE_TITLES = {
    "iid": "IID",
    "dirichlet_03": "Dirichlet alpha=0.3",
    "dirichlet_01": "Dirichlet alpha=0.1",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Stage 2 partition heatmaps.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "extracted")
    parser.add_argument(
        "--partition-dir",
        type=Path,
        default=ROOT / "experiments" / "stage2_fl" / "partitioning",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "experiments" / "stage2_fl" / "results",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=ROOT / "experiments" / "stage2_fl" / "plots",
    )
    return parser.parse_args()


def load_class_names(data_dir):
    with open(data_dir / "cat8_mapping.json") as f:
        mapping = json.load(f)
    id_to_category = {int(v): k for k, v in mapping.items()}
    return [id_to_category[i] for i in range(len(id_to_category))]


def load_labels(data_dir):
    return pd.read_csv(data_dir / "y_train_cat8.csv").squeeze().to_numpy(dtype=np.int32)


def count_partition_classes(partition_path, mode, y_train, class_names):
    npz = np.load(partition_path)
    client_keys = sorted(
        [key for key in npz.files if key.startswith("client_")],
        key=lambda key: int(key.split("_")[1]),
    )

    rows = []
    for key in client_keys:
        client = key
        labels = y_train[npz[key]]
        counts = np.bincount(labels, minlength=len(class_names))
        for class_id, class_name in enumerate(class_names):
            rows.append({
                "mode": mode,
                "client": client,
                "class": class_name,
                "count": int(counts[class_id]),
            })

    npz.close()
    return pd.DataFrame(rows)


def plot_heatmap(counts_df, mode, class_names, output_path):
    clients = sorted(
        counts_df["client"].unique(),
        key=lambda value: int(value.split("_")[1]),
    )
    matrix = (
        counts_df.pivot(index="client", columns="class", values="count")
        .reindex(index=clients, columns=class_names)
        .fillna(0)
        .to_numpy(dtype=float)
    )
    row_totals = matrix.sum(axis=1, keepdims=True)
    fractions = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(fractions, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_title(f"Client Class Distribution - {MODE_TITLES[mode]}")
    ax.set_xlabel("Class")
    ax.set_ylabel("Client")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(clients)))
    ax.set_yticklabels(clients)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fraction of client samples")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(args.data_dir)
    y_train = load_labels(args.data_dir)

    all_counts = []
    plot_paths = []
    for mode, filename in PARTITION_FILES.items():
        partition_path = args.partition_dir / filename
        if not partition_path.exists():
            raise FileNotFoundError(f"Partition file not found: {partition_path}")

        counts = count_partition_classes(partition_path, mode, y_train, class_names)
        all_counts.append(counts)

        plot_path = args.plots_dir / f"partition_heatmap_{mode}.png"
        plot_heatmap(counts, mode, class_names, plot_path)
        plot_paths.append(plot_path)

    counts_raw = pd.concat(all_counts, ignore_index=True)
    counts_path = args.results_dir / "partition_client_class_counts.csv"
    counts_raw.to_csv(counts_path, index=False)

    print("Saved:")
    for path in [counts_path] + plot_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
