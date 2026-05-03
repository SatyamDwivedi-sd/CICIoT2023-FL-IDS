"""Plot multi-seed FedAvg convergence curves for Stage 2.

Reads:
  experiments/stage2_fl/fedavg/fedavg_round_log_{mode}_seed{seed}.json

Writes:
  experiments/stage2_fl/results/fedavg_convergence_raw.csv
  experiments/stage2_fl/results/fedavg_convergence_summary.csv
  experiments/stage2_fl/plots/fedavg_macro_f1_convergence_multiseed.png
  experiments/stage2_fl/plots/fedavg_accuracy_convergence_multiseed.png
  experiments/stage2_fl/plots/fedavg_weighted_f1_convergence_multiseed.png
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODES = ("iid", "dirichlet_03", "dirichlet_01")
MODE_LABELS = {
    "iid": "IID",
    "dirichlet_03": "$\\alpha=0.3$",
    "dirichlet_01": "$\\alpha=0.1$",
}
COLORS = {
    "iid": "steelblue",
    "dirichlet_03": "darkorange",
    "dirichlet_01": "seagreen",
}
METRICS = {
    "f1_macro": ("Macro F1", "fedavg_macro_f1_convergence_multiseed.png"),
    "accuracy": ("Accuracy", "fedavg_accuracy_convergence_multiseed.png"),
    "f1_weighted": ("Weighted F1", "fedavg_weighted_f1_convergence_multiseed.png"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot FedAvg multi-seed convergence.")
    parser.add_argument(
        "--fedavg-dir",
        type=Path,
        default=ROOT / "experiments" / "stage2_fl" / "fedavg",
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


def log_file_pattern():
    mode_alt = "|".join(re.escape(mode) for mode in MODES)
    return re.compile(rf"^fedavg_round_log_(?P<mode>{mode_alt})_seed(?P<seed>\d+)\.json$")


def load_round_log(path):
    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload.get("round_log", [])
    return payload


def load_raw(fedavg_dir):
    pattern = log_file_pattern()
    rows = []

    for path in sorted(fedavg_dir.glob("fedavg_round_log_*_seed*.json")):
        match = pattern.match(path.name)
        if not match:
            continue

        mode = match.group("mode")
        seed = int(match.group("seed"))
        for entry in load_round_log(path):
            rows.append({
                "mode": mode,
                "setting": MODE_LABELS[mode],
                "seed": seed,
                "round": int(entry["round"]),
                "accuracy": float(entry["accuracy"]),
                "f1_macro": float(entry["f1_macro"]),
                "f1_weighted": float(entry["f1_weighted"]),
            })

    if not rows:
        raise FileNotFoundError(
            f"No multi-seed FedAvg round logs found in {fedavg_dir}. "
            "Expected names like fedavg_round_log_iid_seed123.json."
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["mode", "seed", "round"])
    return df


def summarize(raw):
    metric_cols = ["accuracy", "f1_macro", "f1_weighted"]
    grouped = raw.groupby(["mode", "setting", "round"], sort=True)
    summary = grouped[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(part for part in col if part)
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]
    return summary


def plot_metric(summary, metric, ylabel, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for mode in MODES:
        df_mode = summary[summary["mode"] == mode]
        if df_mode.empty:
            continue

        rounds = df_mode["round"].to_numpy()
        mean = df_mode[f"{metric}_mean"].to_numpy()
        std = df_mode[f"{metric}_std"].fillna(0.0).to_numpy()
        color = COLORS[mode]

        ax.plot(
            rounds,
            mean,
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=MODE_LABELS[mode],
            color=color,
        )
        ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_title(f"FedAvg {ylabel} Convergence Across Seeds")
    ax.set_xlabel("Communication round")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.fedavg_dir)
    raw_path = args.results_dir / "fedavg_convergence_raw.csv"
    raw.to_csv(raw_path, index=False)

    summary = summarize(raw)
    summary_path = args.results_dir / "fedavg_convergence_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_paths = []
    for metric, (ylabel, filename) in METRICS.items():
        path = args.plots_dir / filename
        plot_metric(summary, metric, ylabel, path)
        plot_paths.append(path)

    print("Saved:")
    for path in [raw_path, summary_path] + plot_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
