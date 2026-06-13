# SEDE 2026 Submission Snapshot

Paper title:
Impact of Non-IID Heterogeneity on Federated IoT Intrusion Detection: A CICIoT2023 Study

Venue:
SEDE 2026

Submission format:
Springer LNCS

Submission status:
Submitted through Springer Meteor. Waiting for decision.

Main experiments included:
- Centralized baselines: Random Forest, XGBoost, MLP, 1D CNN
- FedAvg MLP under IID, Dirichlet alpha=0.3, and Dirichlet alpha=0.1
- FedAvg 1D CNN architecture-sensitivity experiment
- FedProx supporting baseline with mu=0.01

Federated setup:
- 10 simulated clients
- 20 communication rounds
- 3 local epochs
- batch size 4096
- seeds: 123, 2026, 3407

Note:
Raw CICIoT2023 dataset files, large extracted CSVs, partition index NPZ files, and trained model checkpoint files are not tracked in GitHub.
