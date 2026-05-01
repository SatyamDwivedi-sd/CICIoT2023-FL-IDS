# Stage 2 - FedAvg Results Summary

## Purpose

This summary compares the centralized Stage 1 MLP reference against Stage 2 FedAvg results under IID, Dirichlet 0.3, and Dirichlet 0.1 client partitioning.

## Overall Test Metrics

| Setting | Test Accuracy | Test F1 Macro | Test F1 Weighted |
|---|---:|---:|---:|
| Centralized MLP | 0.7439 | 0.5861 | 0.7604 |
| FedAvg IID | 0.7774 | 0.5053 | 0.7220 |
| FedAvg Dirichlet 0.3 | 0.7967 | 0.5045 | 0.7582 |
| FedAvg Dirichlet 0.1 | 0.7658 | 0.3373 | 0.7084 |

## Key Per-Class F1

| Setting | Web | BruteForce | Spoofing | Benign |
|---|---:|---:|---:|---:|
| Centralized MLP | 0.0710 | 0.2600 | 0.6040 | 0.7830 |
| FedAvg IID | 0.0000 | 0.1930 | 0.3842 | 0.7276 |
| FedAvg Dirichlet 0.3 | 0.0960 | 0.1609 | 0.2024 | 0.7173 |
| FedAvg Dirichlet 0.1 | 0.0171 | 0.0000 | 0.0001 | 0.0000 |

## Runtime

| Setting | Runtime Source | Runtime |
|---|---|---:|
| Centralized MLP | Not included in Stage 2 local files | N/A |
| FedAvg IID | Local notebook run round log | 3943.4 s total |
| FedAvg Dirichlet 0.3 | Saved round log / cluster run | 1457.6 s total |
| FedAvg Dirichlet 0.1 | Saved round log / cluster run | 1478.9 s total |

Runtime totals are computed as the sum of per-round `elapsed_s` values across 20 FedAvg rounds.

## Interpretation

- FedAvg IID improves test accuracy over the centralized MLP reference, but reduces macro F1 from 0.5861 to 0.5053.
- FedAvg Dirichlet 0.3 keeps strong accuracy at 0.7967 and nearly matches the centralized weighted F1, but its macro F1 remains lower than the centralized reference.
- FedAvg Dirichlet 0.1 causes a strong collapse in class-balanced performance, with macro F1 falling to 0.3373 and near-zero F1 for BruteForce, Spoofing, and Benign.
- Accuracy alone is misleading for this problem because high overall accuracy can coexist with poor minority-class or class-balanced performance.

## Sources

- `experiments/stage2_fl/results/fedavg_final_metrics_iid.json`
- `experiments/stage2_fl/results/fedavg_final_metrics_dirichlet_03.json`
- `experiments/stage2_fl/results/fedavg_final_metrics_dirichlet_01.json`
- `experiments/stage2_fl/fedavg/fedavg_round_log_iid.json`
- `experiments/stage2_fl/fedavg/fedavg_round_log_dirichlet_03.json`
- `experiments/stage2_fl/fedavg/fedavg_round_log_dirichlet_01.json`
