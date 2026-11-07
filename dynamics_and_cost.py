"""
dynamics_and_cost.py
====================
Dynamical stability analysis (Lyapunov exponents, perturbation convergence)
and computational cost benchmark (per-beat inference latency, memory footprint).

Matches Section 6.6 and Table 10 of:
"Genetically Tuned Reservoir Computing for Next-Beat ECG Forecasting and Label-Free Anomaly Detection:
A Negative-Findings Benchmark Against Simple Sequence Baselines"
(Arafat Ali Mondal and Ankush Kumar)
"""

import time
import numpy as np
from models import compute_reservoir_states, D


def analyze_reservoir_dynamics(R, data, rids, seed=42, steps=1500):
    """
    Estimates Lyapunov exponents and measures two-copy state perturbation convergence:
      1. Maximum and mean Lyapunov exponents over test sequences.
      2. Share of negative exponents (100% confirms non-chaotic echo state property).
      3. State distance ||x_a(t) - x_b(t)||_2 at 1, 10, 50, 100, 300 beats.
    """
    rng = np.random.default_rng(seed)
    N = R["N"]
    dend = R["dendritic"]
    alpha = R["alpha"]
    W = R["W"]
    Win = R["Win"]
    Wd = R["Wd"]
    c = R["c"]
    gamma = R["gamma"]
    beta = R["beta"]

    lyap_exponents = []

    for rid in rids:
        U = data.U[rid][:steps]
        A = U @ Win.T
        Dd = np.tanh(U @ Wd.T + c) if dend else None

        x = np.zeros(N)
        v = rng.standard_normal(N)
        v /= np.linalg.norm(v)

        eps = 1e-6
        xp = x + eps * v
        accum_log_div = 0.0
        count = 0

        for t in range(len(U)):
            # Update base trajectory
            r = W @ x + A[t]
            if dend:
                gate = 0.5 * (1.0 + np.tanh(0.5 * beta * r))
                cand = np.tanh(r + gamma * Dd[t]) * gate
            else:
                cand = np.tanh(r)
            x = (1.0 - alpha) * x + alpha * cand

            # Update perturbed trajectory
            rp = W @ xp + A[t]
            if dend:
                gate_p = 0.5 * (1.0 + np.tanh(0.5 * beta * rp))
                cand_p = np.tanh(rp + gamma * Dd[t]) * gate_p
            else:
                cand_p = np.tanh(rp)
            xp = (1.0 - alpha) * xp + alpha * cand_p

            dist = np.linalg.norm(xp - x)
            if dist < 1e-300:
                break
            if t >= 50:
                accum_log_div += np.log(dist / eps)
                count += 1
            # Renormalize perturbation
            xp = x + (xp - x) * (eps / dist)

        if count > 0:
            lyap_exponents.append(accum_log_div / count)

    # Two-copy initial state convergence test
    U_lead = data.U[rids[0]][:400]
    A_lead = U_lead @ Win.T
    Dd_lead = np.tanh(U_lead @ Wd.T + c) if dend else None

    xa = rng.normal(0, 0.5, N)
    xb = rng.normal(0, 0.5, N)
    distances = []

    for t in range(len(U_lead)):
        ra = W @ xa + A_lead[t]
        rb = W @ xb + A_lead[t]
        if dend:
            ga = 0.5 * (1.0 + np.tanh(0.5 * beta * ra))
            gb = 0.5 * (1.0 + np.tanh(0.5 * beta * rb))
            canda = np.tanh(ra + gamma * Dd_lead[t]) * ga
            candb = np.tanh(rb + gamma * Dd_lead[t]) * gb
        else:
            canda = np.tanh(ra)
            candb = np.tanh(rb)

        xa = (1.0 - alpha) * xa + alpha * canda
        xb = (1.0 - alpha) * xb + alpha * candb
        distances.append(float(np.linalg.norm(xa - xb)))

    checkpoints = [1, 10, 50, 100, 300]
    distance_at_checkpoints = {
        str(step): distances[step - 1] for step in checkpoints if step <= len(distances)
    }

    return {
        "lyap_mean": float(np.mean(lyap_exponents)),
        "lyap_sd": float(np.std(lyap_exponents)),
        "lyap_max": float(np.max(lyap_exponents)),
        "share_negative": float(np.mean(np.array(lyap_exponents) < 0)),
        "two_copy_distance": distance_at_checkpoints
    }


def benchmark_computational_cost(R, W_out, data, rids, n_samples=1500):
    """
    Measures per-beat inference latency and counts stored parameters / RAM footprint.
    """
    U = np.vstack([data.U[r] for r in rids])[:n_samples]
    N = R["N"]
    dend = R["dendritic"]
    alpha = R["alpha"]
    W = R["W"]
    Win = R["Win"]
    Wd = R["Wd"]
    c = R["c"]
    gamma = R["gamma"]
    beta = R["beta"]

    x = np.zeros(N)
    latencies_ms = []

    for t in range(len(U)):
        u_t = U[t]
        t0 = time.perf_counter()

        # Step calculation
        a = Win @ u_t
        r = W @ x + a
        if dend:
            d = np.tanh(Wd @ u_t + c)
            gate = 0.5 * (1.0 + np.tanh(0.5 * beta * r))
            cand = np.tanh(r + gamma * d) * gate
        else:
            cand = np.tanh(r)
        x = (1.0 - alpha) * x + alpha * cand

        # Linear readout
        x_aug = np.append(x, 1.0)
        _ = x_aug @ W_out

        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1e3)

    # Parameter accounting
    nnz_recurrent = int(np.count_nonzero(W))
    readout_params = int(W_out.size)
    total_stored_floats = (
        readout_params + nnz_recurrent + Win.size + ((Wd.size + c.size) if dend else 0)
    )
    memory_kb = (total_stored_floats * 4.0) / 1024.0  # float32 = 4 bytes

    return {
        "N": N,
        "readout_params": readout_params,
        "recurrent_nonzeros": nnz_recurrent,
        "stored_params": total_stored_floats,
        "memory_kb": memory_kb,
        "latency_ms_mean": float(np.mean(latencies_ms)),
        "latency_ms_p95": float(np.percentile(latencies_ms, 95))
    }

