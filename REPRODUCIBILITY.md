# Reproducibility Guide

This document explains how to reproduce the main experiments and results for the SEDE 2026 submission:

**Impact of Non-IID Heterogeneity on Federated IoT Intrusion Detection: A CICIoT2023 Study**

## 1. Repository Scope

This repository contains:

- experiment scripts
- SLURM job scripts
- summarized result files
- plots and figures
- the submitted SEDE 2026 paper snapshot

This repository does **not** include:

- raw CICIoT2023 dataset files
- extracted or merged CSV files
- large partition index `.npz` files
- trained model checkpoint files
- SLURM output logs

These files are excluded because of size and reproducibility hygiene.

## 2. Environment Setup

The project uses Python 3.10 or newer.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
