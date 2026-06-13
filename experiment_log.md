# Experiment Log

## SEDE 2026 Submitted Snapshot

Paper:
Impact of Non-IID Heterogeneity on Federated IoT Intrusion Detection: A CICIoT2023 Study

Status:
Submitted to SEDE 2026 through Springer Meteor.

## Main Experimental Stages

### Stage 1: Centralized Baselines

Models:
- Random Forest
- XGBoost
- MLP
- 1D CNN

Task:
CICIoT2023 cat8 multiclass intrusion detection.

Evaluation:
Imbalanced holdout test set using accuracy, macro-F1, weighted-F1, and per-class F1.

### Stage 2: FedAvg MLP

Federated setup:
- 10 simulated clients
- 20 communication rounds
- 3 local epochs
- batch size 4096
- seeds: 123, 2026, 3407

Partitions:
- IID
- Dirichlet alpha = 0.3
- Dirichlet alpha = 0.1

### Stage 2B: FedProx

FedProx was evaluated as a supporting heterogeneity-aware baseline using mu = 0.01.

Note:
IID and alpha = 0.3 use three seeds. Alpha = 0.1 uses two completed seeds due to computational/data-loading limitations.

### Architecture Sensitivity: FedAvg 1D CNN

FedAvg-1D CNN was added after advisor feedback to test whether class collapse was specific to the MLP architecture.

All 9 runs completed:
- IID, alpha = 0.3, alpha = 0.1
- seeds: 123, 2026, 3407

## Repository Snapshot Notes

Large raw datasets, extracted CSV files, partition NPZ files, and model checkpoints are not tracked in GitHub. They remain available locally for future journal-extension work.
