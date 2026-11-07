"""
reproduce_paper_results.py
==========================
Rapid reproduction script for:
"Genetically Tuned Reservoir Computing for Next-Beat ECG Forecasting and Label-Free Anomaly Detection:
A Negative-Findings Benchmark Against Simple Sequence Baselines"
(Arafat Ali Mondal and Ankush Kumar)

Uses the paper's exact archived Table 5 genomes (Seeds 42 and 123) to reproduce:
  - Table 6: Next-beat RMSE, ROC-AUC, Average Precision (AP), and paired dRMSE vs baselines
  - Table 7: Paired per-record tests & Wilcoxon p-values
  - Table 8: Thresholded clinical alarm metrics (global & per-patient at q=0.95 and q=0.99)
  - Table 9: Robustness audit excluding noisy records 219 and 231 (n=20)
  - Table 10: Reservoir dynamical stability (Lyapunov exponents, convergence) & computational latency
"""

import os
import sys
import numpy as np
from data_loader import Dataset
from models import (
    predict_persistence, predict_meank, fit_ridge_baseline, build_ridge_features,
    build_reservoir, fit_reservoir_readout, predict_reservoir
)
from genetic_algorithm import ARCHIVED_GENOMES
from evaluate import (
    score_record_predictions, compute_pooled_metrics, evaluate_threshold_alarms,
    compute_bootstrap_ci, run_paired_per_record_tests
)
from dynamics_and_cost import analyze_reservoir_dynamics, benchmark_computational_cost


def main():
    print("=" * 80)
    print(" NERC ECG BENCHMARK: PAPER RESULTS REPRODUCTION")
    print(" Paper: Mondal & Kumar (MIT-BIH Split A, 44 Unpaced Records)")
    print("=" * 80)

    # 1. Load Data
    data = Dataset(split_name="A", beats="annotation")
    test_rids = data.split["test"]
    calib_rids = data.split["calib"]
    val_rids = data.split["val"]

    # 2. Evaluate Baselines
    print("\n[1/5] Evaluating Simple Sequence Baselines...")
    res = {}

    def wrap_predictor(fn):
        return {r: score_record_predictions(fn(r), data.U[r], data.lab[r], data.samp[r], data.fs) for r in sum(data.split.values(), [])}

    res["Persistence"] = wrap_predictor(lambda r: predict_persistence(data.U[r]))

    for k in (2, 3, 5, 10):
        res[f"Mean-of-{k}"] = wrap_predictor(lambda r, k=k: predict_meank(data.U[r], k))

    res["Mean-of-10 (Best)"] = res["Mean-of-10"]

    # Ridge baselines (lag 1 and 3)
    w_ridge1 = fit_ridge_baseline(data, k=1, val_rids=val_rids)
    res["Ridge, 1 beat"] = wrap_predictor(lambda r: build_ridge_features(data.U[r], 1) @ w_ridge1)

    w_ridge3 = fit_ridge_baseline(data, k=3, val_rids=val_rids)
    res["Ridge, 3 beats"] = wrap_predictor(lambda r: build_ridge_features(data.U[r], 3) @ w_ridge3)

    # 3. Evaluate Reservoirs (Seeds 42 and 123 from Table 5)
    print("\n[2/5] Evaluating GA-Tuned Reservoirs (Archived Genomes, Seeds 42 & 123)...")
    res_seeds = {"ESN + GA": [], "Dendritic + GA": []}
    stored_models = {}

    for model_key, is_dend in [("ESN + GA", False), ("Dendritic + GA", True)]:
        arch_key = "dendritic_ga" if is_dend else "esn_ga"
        for seed in [42, 123]:
            genome = ARCHIVED_GENOMES[arch_key][seed]
            R = build_reservoir(genome, seed=seed, dendritic=is_dend)
            W_out = fit_reservoir_readout(R, data)
            parts = wrap_predictor(lambda r, R=R, Wo=W_out: predict_reservoir(R, Wo, data.U[r]))
            res_seeds[model_key].append(parts)
            if seed == 42:
                stored_models[model_key] = (R, W_out)

    # Average across seeds for the reservoirs
    for m in ["ESN + GA", "Dendritic + GA"]:
        # Average SSE and concatenate residuals
        averaged = {}
        for r in sum(data.split.values(), []):
            avg_sse = np.mean([res_seeds[m][s][r]["sse"] for s in range(len(res_seeds[m]))])
            avg_sc = np.mean([res_seeds[m][s][r]["sc"] for s in range(len(res_seeds[m]))], axis=0)
            avg_sc_all = np.mean([res_seeds[m][s][r]["sc_all"] for s in range(len(res_seeds[m]))], axis=0)
            averaged[r] = {
                "sse": avg_sse,
                "n": res_seeds[m][0][r]["n"],
                "sc": avg_sc,
                "lb": res_seeds[m][0][r]["lb"],
                "tt": res_seeds[m][0][r]["tt"],
                "sc_all": avg_sc_all,
                "fs": data.fs
            }
        res[m] = averaged

    # -------------------------------------------------------------------------
    # TABLE 6: Forecasting and Ranking Results on 22 Test Records
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" TABLE 6: Forecasting and Ranking Results on Held-out Test Records (n = 22)")
    print("=" * 80)
    print(f"{'Model':<22} | {'RMSE [95% CI]':<26} | {'ROC-AUC':<9} | {'Avg. Precision':<15} | {'dRMSE vs Persist':<18}")
    print("-" * 100)

    # Record-level bootstrap (50 resamples as documented in Table 6)
    rng = np.random.default_rng(42)
    n_boot = 50
    boot_indices = [rng.integers(0, len(test_rids), len(test_rids)) for _ in range(n_boot)]

    models_order = [
        "Persistence", "Mean-of-2", "Mean-of-3", "Mean-of-5", "Mean-of-10 (Best)",
        "Ridge, 1 beat", "Ridge, 3 beats", "ESN + GA", "Dendritic + GA"
    ]

    table6_data = {}
    for m in models_order:
        test_parts = [res[m][r] for r in test_rids]
        pooled_full = compute_pooled_metrics(test_parts)

        boot_metrics = []
        for b_idx in boot_indices:
            b_parts = [res[m][test_rids[i]] for i in b_idx]
            boot_metrics.append(compute_pooled_metrics(b_parts)["rmse"])

        ci_lo, ci_hi = compute_bootstrap_ci(boot_metrics)
        table6_data[m] = (pooled_full["rmse"], pooled_full["auc"], pooled_full["ap"], ci_lo, ci_hi)

        diff_persist = pooled_full["rmse"] - table6_data["Persistence"][0]
        print(f"{m:<22} | {pooled_full['rmse']:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] | {pooled_full['auc']:.3f}     | {pooled_full['ap']:.3f}           | {diff_persist:+.4f}")

    # -------------------------------------------------------------------------
    # TABLE 7: Paired Per-Record Comparisons
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" TABLE 7: Paired Per-Record Statistical Comparisons (22 Test Records)")
    print("=" * 80)
    print(f"{'Comparison':<35} | {'Delta RMSE [95% CI]':<28} | {'First Lower':<12} | {'Wilcoxon p':<10}")
    print("-" * 92)

    comparisons = [
        ("ESN + GA vs Persistence", "ESN + GA", "Persistence"),
        ("ESN + GA vs Mean-of-10", "ESN + GA", "Mean-of-10 (Best)"),
        ("ESN + GA vs Ridge, 1 beat", "ESN + GA", "Ridge, 1 beat"),
        ("ESN + GA vs Ridge, 3 beats", "ESN + GA", "Ridge, 3 beats"),
        ("Dendritic + GA vs Ridge, 3 beats", "Dendritic + GA", "Ridge, 3 beats"),
        ("Dendritic + GA vs ESN + GA", "Dendritic + GA", "ESN + GA")
    ]

    for label, m_a, m_b in comparisons:
        comp = run_paired_per_record_tests(res[m_a], res[m_b], test_rids, n_bootstrap=1000)
        d_mean = comp["delta_mean"]
        c_lo, c_hi = comp["ci"]
        p_str = "<0.001" if comp["p_value"] < 0.001 else f"{comp['p_value']:.2f}"
        print(f"{label:<35} | {d_mean:+.4f} [{c_lo:+.4f}, {c_hi:+.4f}]   | {comp['first_lower']:<12} | {p_str:<10}")

    # -------------------------------------------------------------------------
    # TABLE 8: Thresholded Alarm Behavior
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" TABLE 8: Thresholded Alarm Behavior on 22 Test Records")
    print("=" * 80)
    print(f"{'Model':<18} | {'Rule':<12} | {'q':<5} | {'Precision':<9} | {'Recall':<8} | {'F1':<8} | {'FA / Hour':<10}")
    print("-" * 80)

    alarm_models = ["Persistence", "Mean-of-10 (Best)", "ESN + GA", "Dendritic + GA"]
    for m in alarm_models:
        test_parts = [res[m][r] for r in test_rids]
        calib_scores = np.concatenate([res[m][r]["sc"] for r in calib_rids])

        for mode in ["global", "per_patient"]:
            for q in [0.95, 0.99]:
                alm = evaluate_threshold_alarms(test_parts, mode=mode, q=q, calib_scores=calib_scores)
                rule_str = "Global" if mode == "global" else "Per-patient"
                print(f"{m:<18} | {rule_str:<12} | {q:<5} | {alm['precision']:<9.3f} | {alm['recall']:<8.3f} | {alm['f1']:<8.3f} | {alm['fa_per_hour']:<10.1f}")

    # -------------------------------------------------------------------------
    # TABLE 9: Robustness Check Without Noisy Records 219 and 231 (n = 20)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" TABLE 9: Robustness Audit Excluding Noisy Records 219 and 231 (n = 20)")
    print("=" * 80)
    print(f"{'Model':<18} | {'RMSE':<8} | {'ROC-AUC':<8} | {'Avg. Precision':<14}")
    print("-" * 55)

    clean_test_rids = [r for r in test_rids if r not in [219, 231]]
    for m in ["Persistence", "Mean-of-10 (Best)", "Ridge, 3 beats", "ESN + GA", "Dendritic + GA"]:
        clean_parts = [res[m][r] for r in clean_test_rids]
        m_clean = compute_pooled_metrics(clean_parts)
        print(f"{m:<18} | {m_clean['rmse']:<8.4f} | {m_clean['auc']:<8.3f} | {m_clean['ap']:<14.3f}")

    # -------------------------------------------------------------------------
    # TABLE 10: Reservoir Dynamics and Computational Cost
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" TABLE 10: Dynamics and Computational Cost (Seed 42 Models)")
    print("=" * 80)

    for m in ["ESN + GA", "Dendritic + GA"]:
        R, W_out = stored_models[m]
        dyn = analyze_reservoir_dynamics(R, data, test_rids[:3], seed=42)
        bench = benchmark_computational_cost(R, W_out, data, test_rids[:3], n_samples=1500)

        print(f"\nModel: {m} (N = {R['N']} units)")
        print(f"  Mean Lyapunov Exponent : {dyn['lyap_mean']:.4f} +/- {dyn['lyap_sd']:.4f}")
        print(f"  Maximum Lyapunov Exp.  : {dyn['lyap_max']:.4f} (All negative: {dyn['share_negative']*100:.0f}%)")
        print(f"  Two-copy distance      : 1 beat: {dyn['two_copy_distance']['1']:.2f} | 100 beats: {dyn['two_copy_distance']['100']:.2e} | 300 beats: {dyn['two_copy_distance']['300']:.2e}")
        print(f"  Stored Parameters      : {bench['stored_params']:,} ({bench['memory_kb']:.1f} KB in float32)")
        print(f"  Inference Latency      : {bench['latency_ms_mean']:.4f} ms/beat (95th pct: {bench['latency_ms_p95']:.4f} ms)")

    print("\n" + "=" * 80)
    print(" REPRODUCTION COMPLETE: ALL PAPER TABLES GENERATED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()

