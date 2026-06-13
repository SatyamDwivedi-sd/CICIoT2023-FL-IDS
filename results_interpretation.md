# Results Interpretation

## Main Finding

The main result is that aggregate accuracy can remain nontrivial while macro-F1 and per-class F1 collapse under strong non-IID label-distribution skew.

This is important for federated IoT intrusion detection because a system may appear acceptable under accuracy while failing on rare or operationally important classes.

## Centralized Baseline Interpretation

The centralized models show that minority classes are already difficult before federated partitioning is introduced. XGBoost performs best overall, but Web and BruteForce remain weak compared with high-support classes such as DDoS, DoS, and Mirai.

## FedAvg Interpretation

FedAvg performs well under IID partitioning, but class-balanced performance degrades as heterogeneity increases.

The key pattern is:

- IID: strongest macro-F1
- Dirichlet alpha = 0.3: moderate degradation
- Dirichlet alpha = 0.1: severe degradation and class collapse

Under alpha = 0.1, BruteForce and Benign collapse to near-zero or zero F1 in the MLP FedAvg setting.

## FedAvg 1D CNN Interpretation

The 1D CNN architecture-sensitivity experiment shows a similar degradation trend. This suggests that the observed class-collapse behavior is not only an MLP artifact.

## FedProx Interpretation

FedProx provides a useful supporting baseline, but proximal regularization alone does not fully solve class-level collapse under strong label skew.

## Practical Implication

Accuracy-only evaluation is not sufficient for FL-based IoT IDS. Multi-seed macro-F1 and per-class F1 should be reported, especially under imbalanced test distributions and non-IID client partitions.
