"""
evaluate.py
===========
Evaluation metrics, alarm thresholding, and record-level bootstrap statistical testing.

Matches Section 3, 5.3, 5.6 and Tables 6, 7, 8 of:
"Genetically Tuned Reservoir Computing for Next-Beat ECG Forecasting and Label-Free Anomaly Detection:
A Negative-Findings Benchmark Against Simple Sequence Baselines"
(Arafat Ali Mondal and Ankush Kumar)
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import wilcoxon
from models import W_OUT, D


def score_record_predictions(pred, U, lab, samp, fs=360):
    """
    Computes per-beat forecasting errors and RMS residuals for a single record:
      Target : U_{t+1} for t >= W_OUT
      Residual anomaly score : e_{t+1} = ||u_{t+1} - u_hat_{t+1}||_2 / sqrt(D)
    """
    tgt = U[W_OUT + 1:]
    e2 = (pred - tgt) ** 2
    sc = np.sqrt(e2.mean(axis=1))  # RMS residual over 100 dimensions

    lb = lab[W_OUT + 1:]
    tt = samp[W_OUT + 1:]
    valid_mask = lb >= 0  # Excludes unlabelled / unclassifiable beats from anomaly metrics

    return {
        "sse": float(e2.sum()),
        "n": int(e2.size),
        "sc": sc[valid_mask],
        "lb": lb[valid_mask],
        "tt": tt[valid_mask],
        "sc_all": sc,
        "fs": fs
    }


def compute_pooled_metrics(record_results):
    """
    Computes pooled test RMSE, ROC-AUC, and Average Precision (AP) across records.
    """
    sse = sum(p["sse"] for p in record_results)
    n = sum(p["n"] for p in record_results)
    rmse = float(np.sqrt(sse / n))

    sc = np.concatenate([p["sc"] for p in record_results])
    lb = np.concatenate([p["lb"] for p in record_results])

    try:
        auc = float(roc_auc_score(lb, sc))
        ap = float(average_precision_score(lb, sc))
    except ValueError:
        auc, ap = np.nan, np.nan

    return {"rmse": rmse, "auc": auc, "ap": ap}


def evaluate_threshold_alarms(record_results, mode="global", q=0.99, calib_scores=None, warmup_beats=300):
    """
    Evaluates clinical alarm behavior at quantile q:
      - Global rule: tau is the q-quantile of calibration residuals.
      - Per-patient adaptive rule: tau is the q-quantile of the first M=300 beats of each record.
    """
    TP = FP = TN = FN = 0
    total_hours = 0.0

    tau_global = np.quantile(calib_scores, q) if mode == "global" and calib_scores is not None else None

    for p in record_results:
        if mode == "global":
            sc, lb, tt = p["sc"], p["lb"], p["tt"]
            tau = tau_global
        else:
            # Per-patient threshold from first warmup_beats
            M = min(warmup_beats, len(p["sc_all"]))
            tau = np.quantile(p["sc_all"][:M], q)
            sc = p["sc"][M:]
            lb = p["lb"][M:]
            tt = p["tt"][M:]

        flagged = sc > tau
        TP += int((flagged & (lb == 1)).sum())
        FP += int((flagged & (lb == 0)).sum())
        FN += int((~flagged & (lb == 1)).sum())
        TN += int((~flagged & (lb == 0)).sum())

        if len(tt) > 1:
            total_hours += (tt[-1] - tt[0]) / p["fs"] / 3600.0

    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    normal_flagged = FP / max(FP + TN, 1)
    fa_per_hour = FP / max(total_hours, 1e-9)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "normal_flagged": normal_flagged,
        "fa_per_hour": fa_per_hour
    }


def compute_bootstrap_ci(values, alpha=0.05):
    """Computes percentile bootstrap 95% confidence interval [2.5%, 97.5%]."""
    arr = np.asarray(values, dtype=float)
    return float(np.nanpercentile(arr, 100 * (alpha / 2.0))), float(np.nanpercentile(arr, 100 * (1.0 - alpha / 2.0)))


def run_paired_per_record_tests(model_a_scores, model_b_scores, test_rids, n_bootstrap=10000, seed=0):
    """
    Performs paired per-record comparison (Table 7):
      - Delta = Mean(Record RMSE A) - Mean(Record RMSE B)
      - 95% record-level bootstrap CI (10,000 resamples)
      - Count of records where model A has lower RMSE than model B
      - Wilcoxon signed-rank test p-value
    """
    rmse_a = np.array([np.sqrt(model_a_scores[r]["sse"] / model_a_scores[r]["n"]) for r in test_rids])
    rmse_b = np.array([np.sqrt(model_b_scores[r]["sse"] / model_b_scores[r]["n"]) for r in test_rids])

    diffs = rmse_a - rmse_b
    delta_mean = float(np.mean(diffs))
    first_lower = int(np.sum(rmse_a < rmse_b))

    # Record-level bootstrap resamples
    rng = np.random.default_rng(seed)
    n = len(test_rids)
    boot_diffs = [np.mean(diffs[rng.integers(0, n, n)]) for _ in range(n_bootstrap)]
    ci_lo, ci_hi = compute_bootstrap_ci(boot_diffs)

    # Wilcoxon signed-rank test
    try:
        res = wilcoxon(rmse_a, rmse_b, alternative="two-sided")
        p_val = float(res.pvalue)
    except Exception:
        p_val = 1.0

    return {
        "delta_mean": delta_mean,
        "ci": (ci_lo, ci_hi),
        "first_lower": f"{first_lower}/{len(test_rids)}",
        "p_value": p_val
    }

