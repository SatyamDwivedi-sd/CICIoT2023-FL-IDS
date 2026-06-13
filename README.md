# CICIoT2023-FL-IDS

This repository contains the code, summarized results, figures, and submission snapshot for the SEDE 2026 paper:

**Impact of Non-IID Heterogeneity on Federated IoT Intrusion Detection: A CICIoT2023 Study**

## Overview

This project studies how non-IID client data distributions affect federated IoT intrusion detection on the CICIoT2023 dataset. The main finding is that aggregate accuracy can remain nontrivial while macro-F1 and per-class F1 reveal severe class-level collapse under strong label-distribution skew.

## Dataset

The raw CICIoT2023 dataset is not included in this repository because of size. The code assumes the dataset is available locally or on the compute cluster.

Large raw downloads, extracted CSV files, partition index files, and trained model checkpoints are intentionally excluded from Git tracking.

## Experiments

The submitted paper includes:

1. Centralized baselines:
   - Random Forest
   - XGBoost
   - MLP
   - 1D CNN

2. Federated learning experiments:
   - FedAvg with MLP
   - FedAvg with 1D CNN architecture-sensitivity experiment
   - FedProx supporting baseline

3. Partition settings:
   - IID
   - Dirichlet alpha = 0.3
   - Dirichlet alpha = 0.1

4. Evaluation:
   - Accuracy
   - Macro-F1
   - Weighted-F1
   - Per-class F1

## Federated Setup

Main configuration:

- 10 simulated clients
- 20 communication rounds
- 3 local epochs
- batch size 4096
- seeds: 123, 2026, 3407
- optimizer: Adam
- learning rate: 1e-4

## Repository Structure

```text
scripts/                            Main experiment and plotting scripts
slurm/                              Cluster job scripts
experiments/stage2_fl/results/      Summary CSV/JSON result files
experiments/stage2_fl/plots/        Figures used for analysis and paper
experiments/stage2_fl/partitioning/ Partition summaries only, not large NPZ index files
doc/submissions/sede2026/           Submitted SEDE 2026 paper snapshot
