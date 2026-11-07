"""
models.py
=========
Implementation of the forecasting models and reservoirs evaluated in the paper:
  - Baselines: Persistence, Moving Average (Mean-of-k), Multi-lag Ridge (1 and 3 beats)
  - Reservoir Computers: Standard Echo State Network (ESN) and Dendritic ESN
  - Linear Ridge Readout with unpenalized intercept

Matches Section 5.2, 5.3, and 5.4 of:
"Genetically Tuned Reservoir Computing for Next-Beat ECG Forecasting and Label-Free Anomaly Detection:
A Negative-Findings Benchmark Against Simple Sequence Baselines"
(Arafat Ali Mondal and Ankush Kumar)
"""

import numpy as np

D = 100
W_OUT = 10  # Washout beats


# -----------------------------------------------------------------------------
# Baseline Predictors
# -----------------------------------------------------------------------------
def predict_persistence(U):
    """Persistence baseline: u_hat_{t+1} = u_t."""
    return U[W_OUT:-1]


def predict_meank(U, k):
    """Moving average baseline: average of previous k beat vectors."""
    cs = np.vstack([np.zeros((1, U.shape[1])), np.cumsum(U, axis=0)])
    t = np.arange(W_OUT, len(U) - 1)
    return (cs[t + 1] - cs[t + 1 - k]) / k


def build_ridge_features(U, k):
    """Builds lag-k feature matrix with an intercept column for linear regression."""
    t = np.arange(W_OUT, len(U) - 1)
    feats = [U[t - j] for j in range(k)] + [np.ones((len(t), 1))]
    return np.hstack(feats)


def fit_ridge_baseline(data, k, val_rids):
    """
    Fits linear ridge regression on previous k beats across training records.
    Selects regularisation parameter lambda on validation records.
    """
    X_tr = np.vstack([build_ridge_features(data.U[r], k) for r in data.split["train"]])
    Y_tr = np.vstack([data.U[r][W_OUT + 1:] for r in data.split["train"]])

    best_val_mse = float("inf")
    best_weights = None

    for lam in 10.0 ** np.arange(-8, 1):
        L = lam * np.eye(X_tr.shape[1])
        L[-1, -1] = 0.0  # Intercept unpenalized
        try:
            Wr = np.linalg.solve(X_tr.T @ X_tr + L, X_tr.T @ Y_tr)
        except np.linalg.LinAlgError:
            Wr = np.linalg.lstsq(X_tr.T @ X_tr + L, X_tr.T @ Y_tr, rcond=None)[0]

        val_mse = np.mean([
            np.mean((build_ridge_features(data.U[r], k) @ Wr - data.U[r][W_OUT + 1:]) ** 2)
            for r in val_rids
        ])

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_weights = Wr

    return best_weights


# -----------------------------------------------------------------------------
# Reservoir Computers (Standard and Dendritic ESN)
# -----------------------------------------------------------------------------
def build_reservoir(genome, seed=42, dendritic=False):
    """
    Constructs reservoir weight matrices:
      W   : Random sparse recurrent matrix scaled to spectral radius rho
      Win : Input matrix scaled by s
      Wd  : Dendritic input matrix ~ N(0, 0.25^2)
      c   : Dendritic bias vector ~ N(0, 0.1^2)
    Note: For a given seed, standard and dendritic models share W, Win, Wd, and c.
    """
    rng = np.random.default_rng(seed)
    N = int(genome["N"])

    # Sparse recurrent matrix
    mask = rng.random((N, N)) < genome["kappa"]
    W0 = rng.standard_normal((N, N)) * mask
    eigvals = np.abs(np.linalg.eigvals(W0))
    max_ev = eigvals.max() if len(eigvals) > 0 else 1.0
    W = W0 * (genome["rho"] / max_ev) if max_ev > 1e-12 else W0

    # Input and dendritic projections
    Win = rng.uniform(-genome["s"], genome["s"], (N, D))
    Wd = rng.normal(0, 0.25, (N, D))
    c = rng.normal(0, 0.1, N)

    return {
        "N": N,
        "W": W,
        "Win": Win,
        "Wd": Wd,
        "c": c,
        "alpha": genome["alpha"],
        "gamma": genome.get("gamma", 0.0),
        "beta": genome.get("beta", 1.0),
        "dendritic": dendritic,
        "lambda": genome["lam"]
    }


def compute_reservoir_states(R, U, x0=None):
    """
    Drives the reservoir state beat by beat through time:
      Standard ESN:
        r_t = W x_{t-1} + Win u_t
        x_t = (1 - alpha) x_{t-1} + alpha tanh(r_t)
      Dendritic ESN:
        d_t = tanh(Wd u_t + c)
        g_t = sigmoid(beta * r_t)
        x_t = (1 - alpha) x_{t-1} + alpha [tanh(r_t + gamma * d_t) * g_t]
    """
    N = R["N"]
    A = U @ R["Win"].T
    dend = R["dendritic"]
    Dd = np.tanh(U @ R["Wd"].T + R["c"]) if dend else None

    alpha = R["alpha"]
    W = R["W"]
    gamma = R["gamma"]
    beta = R["beta"]

    x = np.zeros(N) if x0 is None else x0.copy()
    X = np.empty((len(U), N))

    for t in range(len(U)):
        r = W @ x + A[t]
        if dend:
            gate = 0.5 * (1.0 + np.tanh(0.5 * beta * r))  # numerically stable sigmoid
            candidate = np.tanh(r + gamma * Dd[t]) * gate
        else:
            candidate = np.tanh(r)
        x = (1.0 - alpha) * x + alpha * candidate
        X[t] = x

    return X


def fit_ridge_readout(X, Y, lam):
    """
    Fits linear ridge readout with unpenalized intercept:
      W_out = (X_aug^T X_aug + lam * I_0)^(-1) X_aug^T Y
    """
    Xa = np.hstack([X, np.ones((len(X), 1))])
    L = lam * np.eye(Xa.shape[1])
    L[-1, -1] = 0.0  # unpenalized bias

    try:
        W_out = np.linalg.solve(Xa.T @ Xa + L, Xa.T @ Y)
    except np.linalg.LinAlgError:
        W_out = np.linalg.lstsq(Xa.T @ Xa + L, Xa.T @ Y, rcond=None)[0]

    return W_out


def fit_reservoir_readout(R, data):
    """Fits reservoir readout on all training records after 10-beat washout."""
    X_list, Y_list = [], []
    for rid in data.split["train"]:
        U = data.U[rid]
        X = compute_reservoir_states(R, U)
        X_list.append(X[W_OUT:-1])
        Y_list.append(U[W_OUT + 1:])

    X_train = np.vstack(X_list)
    Y_train = np.vstack(Y_list)
    return fit_ridge_readout(X_train, Y_train, R["lambda"])


def predict_reservoir(R, W_out, U):
    """Predicts next beat u_hat_{t+1} using states and readout."""
    X = compute_reservoir_states(R, U)
    Xa = np.hstack([X[W_OUT:-1], np.ones((len(X) - W_OUT - 1, 1))])
    return Xa @ W_out

