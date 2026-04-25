# Partition Interpretation

## IID
Each of 10 clients receives ~1.47M samples with nearly identical 
class distributions. DDoS ~58%, Web ~0.11%, BruteForce ~0.06% 
per client. This is the idealized baseline — no heterogeneity.

## Dirichlet alpha=0.3 (moderate heterogeneity)
Client sizes vary from 379K to 3.98M samples. Some clients 
specialize — client_2 has 3.4M DDoS but zero DoS. Several 
clients have zero samples for Web or BruteForce. Moderate 
skew that reflects realistic IoT device diversity.

## Dirichlet alpha=0.1 (strong heterogeneity)
Client sizes vary from 46K to 7.67M samples. client_2 holds 
over half the entire training set. Multiple clients have zero 
samples for 3-4 classes. Several clients never see Web or 
BruteForce attacks. This extreme skew will significantly 
degrade minority class detection in FL.

## Key observation
As alpha decreases, minority classes (Web, BruteForce) become 
increasingly concentrated in fewer clients. Under alpha=0.1, 
most clients have zero BruteForce samples — meaning the global 
model receives almost no gradient signal for BruteForce from 
most clients each round. This directly motivates why non-IID 
FL is harder and why per-client threshold detection matters.
