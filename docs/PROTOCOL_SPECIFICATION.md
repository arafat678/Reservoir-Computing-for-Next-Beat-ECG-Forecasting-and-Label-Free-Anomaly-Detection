# Protocol Specification: NERC ECG Benchmark

This document details the exact partitioning, preprocessing, mathematical formulation, and hyperparameter search protocol used in the benchmark.

---

## 1. Patient-Level Data Partitioning (AAMI Split A)

To completely eliminate data leakage and ensure realistic inter-patient generalization, the 44 unpaced records of the **MIT-BIH Arrhythmia Database** (Goldberger et al., 2000; Moody & Mark, 2001) are partitioned strictly at the patient record level:

| Partition | Record Count | Clinical Role & Usage | Record Identifiers |
|:---|:---:|:---|:---|
| **Training Cohort** | 16 records | Fits feature normalization bounds and linear ridge readout weights ($W_{\text{out}}$). | `101, 106, 109, 114, 115, 116, 118, 119, 124, 201, 203, 205, 207, 209, 215, 223` |
| **Validation Cohort** | 3 records | Evaluates Genetic Algorithm objective fitness ($J = \text{MSE}_{\text{val}}$) without diagnostic labels. | `122, 208, 220` |
| **Calibration Cohort** | 3 records | Selects operational alarm threshold quantiles ($\tau = Q_q$). | `112, 113, 121` |
| **Sequestered Test Cohort** | 22 records | Frozen final evaluation cohort; scored strictly once under reporting metrics. | `100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234` |

*(Note: Records 102, 104, 107, 217 contain pacemaker artifacts and are excluded following standard AAMI guidelines).*

---

## 2. Signal Preprocessing & 100-D Beat Representation

1. **Bandpass Filtering:**  
   Raw Lead II ECG signals ($f_s = 360\text{ Hz}$) pass through a zero-phase 4th-order Butterworth bandpass filter ($0.5\text{ Hz} - 40.0\text{ Hz}$) to remove baseline wander and powerline interference.
2. **Morphological Beat Segmentation:**  
   Beats are segmented within a symmetric $\pm 150\text{ ms}$ window (54 samples at $360\text{ Hz}$) centered at the annotated or detected R-peak.
3. **Dual-Domain Feature Vector ($D = 100$):**
   - **50-point Resampled Waveform:** Linear interpolation of time-domain amplitude across 50 uniform intervals.
   - **50-point Spectral Magnitude:** First 50 coefficients of the Discrete Fourier Transform (DFT) magnitude spectrum capturing frequency distribution.
4. **Train-Only Min-Max Scaling:**  
   Features are normalized using training set extrema and clipped to $[0, 1]$:
   $$\mathbf{u}_t = \text{clip}\left(\frac{\mathbf{u}_t^{\text{raw}} - \mathbf{a}}{\mathbf{b} - \mathbf{a}}, 0, 1\right), \quad \mathbf{a} = \min_{t \in \text{Train}}(\mathbf{u}_t^{\text{raw}}), \quad \mathbf{b} = \max_{t \in \text{Train}}(\mathbf{u}_t^{\text{raw}})$$

---

## 3. Mathematical Formulations of Forecasters

### A. Baselines
- **Persistence:** $\hat{\mathbf{u}}_{t+1} = \mathbf{u}_t$
- **Mean-of-Last-$k$:** $\hat{\mathbf{u}}_{t+1} = \frac{1}{k} \sum_{i=0}^{k-1} \mathbf{u}_{t-i}, \quad k \in \{2, 3, 5, 10\}$
- **Autoregressive Linear Ridge:**  
  $$\hat{\mathbf{u}}_{t+1} = [\mathbf{u}_t, \dots, \mathbf{u}_{t-L+1}, 1] \mathbf{W}_{\text{ridge}}, \quad \mathbf{W}_{\text{ridge}} = (\mathbf{Z}^T \mathbf{Z} + \lambda \mathbf{I}_0)^{-1} \mathbf{Z}^T \mathbf{Y}$$

### B. Standard Echo State Network (ESN)
Given recurrent state $\mathbf{x}_t \in \mathbb{R}^N$ (washout period $W = 10$ beats):
$$\mathbf{r}_t = \mathbf{W} \mathbf{x}_{t-1} + \mathbf{W}_{\text{in}} \mathbf{u}_t$$
$$\mathbf{x}_t = (1 - \alpha) \mathbf{x}_{t-1} + \alpha \tanh(\mathbf{r}_t)$$
where $\mathbf{W}$ is scaled to spectral radius $\rho$ with density $\kappa$, and $\mathbf{W}_{\text{in}} \sim s \cdot U(-1, 1)$.

### C. Dendritic Echo State Network
Incorporates an auxiliary non-linear projection and a multiplicative gate:
$$\mathbf{d}_t = \tanh(\mathbf{W}_d \mathbf{u}_t + \mathbf{c})$$
$$\mathbf{g}_t = \sigma(\beta \mathbf{r}_t)$$
$$\mathbf{x}_t = (1 - \alpha) \mathbf{x}_{t-1} + \alpha \left[ \tanh(\mathbf{r}_t + \gamma \mathbf{d}_t) \odot \mathbf{g}_t \right]$$
where $\odot$ denotes the Hadamard (element-wise) product, $\gamma$ governs dendritic coupling, and $\beta$ controls sigmoid gating steepness.

### D. Linear Readout Training
$$\mathbf{W}_{\text{out}} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}_0)^{-1} \mathbf{X}^T \mathbf{Y}$$
where $\mathbf{I}_0 = \text{diag}(1, \dots, 1, 0)$ leaves the intercept unpenalized.

---

## 4. Genetic Algorithm Search Configuration

Hyperparameters are evolved across the validation set using tournament selection, uniform crossover, and gaussian mutation:

| Parameter | Symbol | Search Range | Mutation Std ($\sigma$) | Description |
|:---|:---:|:---:|:---:|:---|
| Reservoir Size | $N$ | $[50, 300]$ (integers) | $15$ | Number of internal recurrent nodes |
| Spectral Radius | $\rho$ | $[0.1, 1.2]$ | $0.05$ | Maximum eigenvalue of recurrent matrix $\mathbf{W}$ |
| Recurrent Density | $\kappa$ | $[0.02, 0.30]$ | $0.02$ | Sparsity level of recurrent connections |
| Input Scaling | $s$ | $[0.01, 1.00]$ | $0.05$ | Scaling multiplier for input projections $\mathbf{W}_{\text{in}}$ |
| Leaking Rate | $\alpha$ | $[0.1, 1.0]$ | $0.05$ | State retention coefficient |
| Ridge Penalty | $\log_{10} \lambda$ | $[-6, +2]$ | $0.5$ | Tikhonov regularization exponent |
| Dendritic Coupling | $\gamma$ | $[0.0, 1.5]$ | $0.1$ | Coupling weight for dendritic branch $\mathbf{d}_t$ |
| Gate Slope | $\beta$ | $[0.1, 2.0]$ | $0.1$ | Sigmoid saturation scaling for $\mathbf{g}_t$ |
