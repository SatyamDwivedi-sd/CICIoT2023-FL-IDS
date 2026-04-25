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

## Status
In progress

## Notebooks
- 09_FL_Partitioning.ipynb — client data partitioning (local)
- 10_FL_FedAvg.ipynb — FedAvg simulation and evaluation (cluster)

## Results location
- partitioning/ — client partition indices and metadata
- fedavg/ — FedAvg round logs
- results/ — final evaluation CSVs
- plots/ — visualizations
- metadata/ — run configs, seeds, and experiment summaries
