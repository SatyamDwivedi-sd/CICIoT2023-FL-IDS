"""Summarize multi-seed FedAvg final metrics for Stage 2.

Reads:
  experiments/stage2_fl/results/fedavg_final_metrics_{mode}_seed{seed}.json

Writes:
  experiments/stage2_fl/results/fedavg_multiseed_raw.csv
  experiments/stage2_fl/results/fedavg_multiseed_summary.csv
  experiments/stage2_fl/results/fedavg_multiseed_paper_table.md
  experiments/stage2_fl/results/fedavg_multiseed_key_classes_raw.csv
  experiments/stage2_fl/results/fedavg_multiseed_key_classes_summary.csv
  experiments/stage2_fl/results/fedavg_multiseed_key_classes_paper_table.md
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODES = ("iid", "dirichlet_03", "dirichlet_01")
MODE_LABELS = {
    "iid": "FedAvg IID",
    "dirichlet_03": "FedAvg α=0.3",
    "dirichlet_01": "FedAvg α=0.1",
}
PRIMARY_METRICS = ("accuracy", "f1_macro", "f1_weighted")
PAYLOAD_METRICS = {
    "accuracy": "test_accuracy",
    "f1_macro": "test_f1_macro",
    "f1_weighted": "test_f1_weighted",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize FedAvg multi-seed metrics.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "experiments" / "stage2_fl" / "results",
    )
    parser.add_argument(
        "--key-classes",
        nargs="+",
        default=["Web", "BruteForce", "Spoofing", "Benign"],
        help="Per-class F1 columns to highlight in key-class outputs.",
    )
    return parser.parse_args()


def metric_file_pattern():
    mode_alt = "|".join(re.escape(mode) for mode in MODES)
    return re.compile(rf"^fedavg_final_metrics_(?P<mode>{mode_alt})_seed(?P<seed>\d+)\.json$")


def extract_per_class_f1(payload):
    if isinstance(payload.get("test_f1_per_class"), dict):
        return {
            f"f1_{class_name}": float(score)
            for class_name, score in payload["test_f1_per_class"].items()
        }

    return {
        key: float(value)
        for key, value in payload.items()
        if key.startswith("f1_") and isinstance(value, (int, float))
    }


def load_rows(results_dir):
    pattern = metric_file_pattern()
    rows = []

    for path in sorted(results_dir.glob("fedavg_final_metrics_*_seed*.json")):
        match = pattern.match(path.name)
        if not match:
            continue

        with open(path) as f:
            payload = json.load(f)

        mode = match.group("mode")
        seed = int(match.group("seed"))
        row = {
            "mode": mode,
            "setting": MODE_LABELS[mode],
            "seed": seed,
        }

        for out_metric, payload_metric in PAYLOAD_METRICS.items():
            row[out_metric] = (
                float(payload[payload_metric])
                if payload_metric in payload
                else pd.NA
            )

        row.update(extract_per_class_f1(payload))
        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No multi-seed FedAvg metric files found in {results_dir}. "
            "Expected names like fedavg_final_metrics_iid_seed123.json."
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["mode", "seed"])
    return df


def summarize(df, metric_cols):
    grouped = df.groupby(["mode", "setting"], sort=True)
    summary = grouped[metric_cols].agg(["mean", "std", "min", "max", "count"]).reset_index()
    summary.columns = [
        "_".join(part for part in col if part)
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]
    return summary


def mean_std_cell(row, metric):
    mean = row.get(f"{metric}_mean", pd.NA)
    std = row.get(f"{metric}_std", pd.NA)
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def write_markdown_table(rows, columns, path):
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n")


def write_primary_paper_table(summary, path):
    table_rows = []
    for _, row in summary.iterrows():
        table_rows.append({
            "Setting": row["setting"],
            "Accuracy": mean_std_cell(row, "accuracy"),
            "F1 Macro": mean_std_cell(row, "f1_macro"),
            "F1 Weighted": mean_std_cell(row, "f1_weighted"),
            "n": int(row["accuracy_count"]),
        })

    write_markdown_table(
        table_rows,
        ["Setting", "Accuracy", "F1 Macro", "F1 Weighted", "n"],
        path,
    )


def write_key_class_paper_table(summary, key_classes, path):
    columns = ["Setting"] + key_classes
    table_rows = []
    for _, row in summary.iterrows():
        out = {"Setting": row["setting"]}
        for class_name in key_classes:
            out[class_name] = mean_std_cell(row, class_name)
        table_rows.append(out)

    write_markdown_table(table_rows, columns, path)


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    raw = load_rows(args.results_dir)
    raw_path = args.results_dir / "fedavg_multiseed_raw.csv"
    primary_raw = raw[["mode", "setting", "seed", *PRIMARY_METRICS]]
    primary_raw.to_csv(raw_path, index=False)

    summary = summarize(primary_raw, list(PRIMARY_METRICS))
    summary_path = args.results_dir / "fedavg_multiseed_summary.csv"
    summary.to_csv(summary_path, index=False)

    paper_path = args.results_dir / "fedavg_multiseed_paper_table.md"
    write_primary_paper_table(summary, paper_path)

    key_raw = raw[["mode", "setting", "seed"]].copy()
    for class_name in args.key_classes:
        source_col = f"f1_{class_name}"
        if source_col in raw.columns:
            key_raw[class_name] = raw[source_col]

    available_key_classes = [
        class_name
        for class_name in args.key_classes
        if class_name in key_raw.columns
    ]
    if not available_key_classes:
        raise ValueError(
            "None of the requested key classes were found in the metric files: "
            + ", ".join(args.key_classes)
        )

    key_raw_path = args.results_dir / "fedavg_multiseed_key_classes_raw.csv"
    key_raw.to_csv(key_raw_path, index=False)

    key_summary = summarize(key_raw, available_key_classes)
    key_summary_path = args.results_dir / "fedavg_multiseed_key_classes_summary.csv"
    key_summary.to_csv(key_summary_path, index=False)

    key_paper_path = args.results_dir / "fedavg_multiseed_key_classes_paper_table.md"
    write_key_class_paper_table(key_summary, available_key_classes, key_paper_path)

    print("Saved:")
    for path in [
        raw_path,
        summary_path,
        paper_path,
        key_raw_path,
        key_summary_path,
        key_paper_path,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
