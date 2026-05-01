# Stage 2 — Federated Learning Simulation

## Current goal
Non-IID partitioning and FedAvg baseline on CICIoT2023 cat8

## Initial configuration
- Dataset: CICIoT2023 cat8 (8-category)
- Model: MLP
- Clients: 10
- Dirichlet alphas: 0.1 and 0.3 (plus IID baseline)
- FL rounds: 20
- Local epochs: 3
- Batch size: 4096
- Random seed: 42

## Status
Complete (current baseline scope)

## Notebooks
- 09_FL_Partitioning.ipynb — client data partitioning
- 10_FL_FedAvg.ipynb — FedAvg simulation and evaluation

## Results location
- partitioning/ — client partition indices and metadata
- fedavg/ — FedAvg round logs
- results/ — final evaluation JSON/CSV files
- plots/ — visualizations
- metadata/ — run configs, seeds, and experiment summaries

## Results Summary
Evaluated on the original imbalanced cat8 test set (4,201,031 samples).

| Setting | Accuracy | F1 Macro | F1 Weighted | Web F1 | BruteForce F1 |
|---|---:|---:|---:|---:|---:|
| Centralized MLP | 0.7439 | 0.5861 | 0.7604 | 0.071 | 0.260 |
| FL IID | 0.7774 | 0.5053 | 0.7220 | 0.000 | 0.193 |
| FL α=0.3 | 0.7967 | 0.5045 | 0.7582 | 0.096 | 0.161 |
| FL α=0.1 | 0.7658 | 0.3373 | 0.7084 | 0.017 | 0.000 |

## Key findings
- FedAvg under IID already underperforms the centralized MLP baseline on macro F1.
- Accuracy alone is misleading in this problem.
- FL α=0.3 achieves the highest accuracy, but macro F1 remains well below the centralized baseline.
- FL α=0.1 causes strong collapse in class-balanced performance.
- Minority classes such as Web and BruteForce are disproportionately harmed as heterogeneity increases.
- These findings motivate Stage 3: benign client behavior is already highly heterogeneous under strong non-IID conditions, making robust attack/defense analysis necessary.

## Runtime note
- FedAvg IID runtime came from the local notebook run.
- Dirichlet α=0.3 and α=0.1 runtimes came from the saved round logs / cluster runs.
