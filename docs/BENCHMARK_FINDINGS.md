# Empirical Findings & Scientific Analysis: NERC ECG Benchmark

This document provides a comprehensive report of the empirical findings obtained under the **strict patient-level evaluation protocol (AAMI Split A, 22 sequestered test patients)**. 

---

## 1. Executive Summary: Core Scientific Questions

| Research Question | Scientific Hypothesis | Empirical Finding | Status |
|:---|:---|:---|:---:|
| **RQ1: Forecasting Advantage** | Does a GA-tuned Echo State Network (ESN) outperform standard linear sequence baselines in next-beat ECG prediction? | **Negative Finding:** While the GA-tuned reservoir achieves a 38.9% RMSE reduction over persistence ($0.1309$ vs $0.2141$), it provides **no statistically significant improvement** over a simple 3-beat linear ridge regression model ($0.1305 \pm 0.0076$, $\Delta\text{RMSE} = -0.0004$, $p = 0.582$). | **Refuted** |
| **RQ2: Bio-Inspired Dendritic Ablation** | Does adding bio-inspired dendritic non-linear projections ($d_t$) and multiplicative sigmoid gating ($g_t$) improve reservoir expressivity? | **Negative Finding:** Under identical evolutionary search budgets and seed alignments, the dendritic ESN matches standard ESN performance exactly ($0.1311$ vs $0.1309$, paired difference includes zero, $p = 0.824$). Multiplicative gating restricts dynamic range without adding predictive value. | **Refuted** |
| **RQ3: Label-Free Anomaly Detection & Alarms** | Can forecasting residuals act as effective anomaly rankings without training labels, while maintaining tolerable false alarm burdens? | **Positive & Nuanced:** Forecasting residuals achieve strong ranking (ROC-AUC $0.965 - 0.969$, PR-AUC $0.865 - 0.872$). However, a 10-beat moving average matches or exceeds reservoir ranking ($1.000$ ROC-AUC, $0.999$ PR-AUC). Under clinical threshold calibration, calibrated global thresholds achieve $\approx 3.5\text{ false alarms/hour}$. | **Confirmed with Baseline Caveat** |

---

## 2. Next-Beat Forecasting & Anomaly Ranking Benchmark

Evaluated strictly across **$n = 22$ held-out test patients** (record-level bootstrap, 50 resamples):

| Model Architecture | Next-Beat RMSE [95% CI] | ROC-AUC [95% CI] | Average Precision (PR-AUC) | $\Delta\text{RMSE}$ vs. Persistence | $\Delta\text{AUC}$ vs. Persistence | Records Model < Persist |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Persistence ($\hat{u}_{t+1} = u_t$)** | 0.2141 [0.1936, 0.2288] | 0.884 [0.872, 0.894] | 0.472 [0.467, 0.479] | $+0.0000$ | $+0.000$ | — |
| **Mean-of-2 Beats** | 0.1852 [0.1688, 0.1983] | 0.960 [0.956, 0.965] | 0.804 [0.788, 0.823] | $-0.0289$ | $+0.076$ | 22/22 |
| **Mean-of-3 Beats** | 0.1748 [0.1592, 0.1872] | 0.983 [0.981, 0.986] | 0.904 [0.886, 0.924] | $-0.0393$ | $+0.099$ | 22/22 |
| **Mean-of-5 Beats** | 0.1657 [0.1510, 0.1775] | 0.996 [0.995, 0.997] | 0.973 [0.963, 0.984] | $-0.0485$ | $+0.112$ | 22/22 |
| **Mean-of-10 Beats (Best MA)** | 0.1580 [0.1444, 0.1690] | **1.000 [1.000, 1.000]** | **0.999 [0.997, 1.000]** | $-0.0561$ | $+0.116$ | 22/22 |
| **Linear Ridge (Lag-1)** | 0.1354 [0.1253, 0.1434] | 0.989 [0.985, 0.992] | 0.935 [0.918, 0.954] | $-0.0787$ | $+0.105$ | 22/22 |
| **Linear Ridge (Lag-3)** | **0.1305 [0.1221, 0.1374]** | 0.974 [0.964, 0.982] | 0.883 [0.851, 0.918] | **-0.0836** | $+0.090$ | 22/22 |
| **GA-Tuned Standard ESN** | 0.1309 [0.1231, 0.1372] | 0.965 [0.959, 0.973] | 0.865 [0.842, 0.889] | $-0.0832$ | $+0.081$ | 22/22 |
| **GA-Tuned Dendritic ESN** | 0.1311 [0.1231, 0.1372] | 0.969 [0.965, 0.975] | 0.872 [0.857, 0.894] | $-0.0831$ | $+0.085$ | 22/22 |

---

## 3. Paired Statistical Significance (Wilcoxon Signed-Rank Tests)

To establish whether the reservoir improvements over baselines are statistically defensible:

- **ESN vs. Persistence:**  
  $\text{Mean }\Delta\text{RMSE} = -0.0832$ ($p = 1.9 \times 10^{-5}$). Decisive improvement over naive tracking across all 22 patients.
- **ESN vs. Lag-3 Ridge Regression:**  
  $\text{Mean }\Delta\text{RMSE} = +0.0004$ ($p = 0.582$). The null hypothesis cannot be rejected; the recurrent reservoir does not outperform a 3-beat autoregressive linear filter.
- **Dendritic ESN vs. Standard ESN:**  
  $\text{Mean }\Delta\text{RMSE} = +0.0002$ ($p = 0.824$). Dendritic compartmentalization yields no empirical advantage.

---

## 4. Clinical Alarm Burden & Thresholding Analysis

Ambulatory telemetry requires minimizing false alarms to prevent nurse desensitization ("alarm fatigue", Sendelbach & Funk, 2013). Thresholds $\tau$ are calibrated strictly on the **3 calibration records** at target percentiles $q \in \{0.95, 0.99\}$:

| Threshold Protocol | Calibrated Quantile $q$ | Precision | Recall (Sensitivity) | False Alarm Burden |
|:---|:---:|:---:|:---:|:---:|
| **Global Calibration** | $q = 0.95$ | 0.482 | 0.764 | 14.8 alarms / hr |
| **Global Calibration** | $q = 0.99$ | **0.718** | **0.583** | **3.5 alarms / hr** |
| **Per-Patient Calibration** | $q = 0.95$ | 0.395 | 0.821 | 22.1 alarms / hr |
| **Per-Patient Calibration** | $q = 0.99$ | 0.641 | 0.694 | 7.2 alarms / hr |

**Key Takeaway:** A global residual threshold chosen at the 99th calibration percentile suppresses false alarms to **$3.5\text{ alarms/hour}$** while capturing nearly $60\%$ of morphologically aberrant beats without any supervised diagnostic labels.

---

## 5. Dynamical Contractivity & Edge Compute Profile

To guarantee numerical stability and real-time execution feasibility on wearable bio-monitors:

- **Empirical Lyapunov Exponents:**  
  Calculated via the Benettin two-trajectory method ($\Delta_0 = 10^{-8}$):
  - Standard ESN: $\lambda_{\max} = -0.103 \pm 0.012$ (100% strictly contractive, negative across all records)
  - Dendritic ESN: $\lambda_{\max} = -0.094 \pm 0.010$ (100% strictly contractive)
  - *Implication:* Fading memory and Echo State Property are strictly preserved; the reservoir is immune to runaway oscillations.
- **Computational Latency:**
  - Per-beat forward pass latency: **$0.138\text{ ms} \pm 0.014\text{ ms}$** (Single CPU thread)
  - Memory footprint: $< 4.2\text{ MB}$ RAM
  - *Implication:* Easily deployable on Cortex-M4 or low-power RISC-V wearable hardware sampling at $360\text{ Hz}$.
