"""
run_benchmark.py
================
Full command-line interface for the NERC ECG Benchmark:
"Genetically Tuned Reservoir Computing for Next-Beat ECG Forecasting and Label-Free Anomaly Detection:
A Negative-Findings Benchmark Against Simple Sequence Baselines"
(Arafat Ali Mondal and Ankush Kumar)

Usage Examples:
  1. Fast reproduction (using archived Table 5 genomes):
     python run_benchmark.py --mode reproduce --out results_reproduce

  2. Full neuroevolution training from scratch:
     python run_benchmark.py --mode ga --pop 30 --gens 40 --seeds 42 123 --out results_ga
"""

import os
import argparse
import json
import time
import numpy as np
from data_loader import Dataset
from models import (
    predict_persistence, predict_meank, fit_ridge_baseline, build_ridge_features,
    build_reservoir, fit_reservoir_readout, predict_reservoir
)
from genetic_algorithm import ARCHIVED_GENOMES, run_genetic_algorithm
from evaluate import (
    score_record_predictions, compute_pooled_metrics, evaluate_threshold_alarms,
    compute_bootstrap_ci, run_paired_per_record_tests
)
from dynamics_and_cost import analyze_reservoir_dynamics, benchmark_computational_cost


def main():
    parser = argparse.ArgumentParser(description="NERC ECG Benchmark CLI Runner")
    parser.add_argument("--mode", default="reproduce", choices=["reproduce", "ga"],
                        help="'reproduce' uses archived paper genomes; 'ga' runs evolutionary search.")
    parser.add_argument("--split", default="A", choices=["A"], help="Dataset split (Split A).")
    parser.add_argument("--beats", default="annotation", choices=["annotation", "xqrs"],
                        help="Beat detection source: cardiologist annotations or Pan-Tompkins xqrs.")
    parser.add_argument("--pop", type=int, default=30, help="GA population size (default 30).")
    parser.add_argument("--gens", type=int, default=40, help="GA generations (default 40).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123],
                        help="Random seeds to evaluate (default: 42 123).")
    parser.add_argument("--boot", type=int, default=50, help="Bootstrap resamples for CIs (default 50).")
    parser.add_argument("--out", default="output", help="Directory to save JSON and markdown summaries.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()

    print("=" * 80)
    print(" NERC RESERVOIR FORECASTING & ANOMALY BENCHMARK")
    print(f" Mode: {args.mode.upper()} | Split: {args.split} | Beats: {args.beats} | Seeds: {args.seeds}")
    print("=" * 80)

    # 1. Load Data
    data = Dataset(split_name=args.split, beats=args.beats)
    test_rids = data.split["test"]
    calib_rids = data.split["calib"]
    val_rids = data.split["val"]

    # 2. Evaluate Baselines
    print("\n[1/4] Fitting and Evaluating Baselines (Persistence, Mean-of-k, Ridge)...")
    res = {}
    all_rids = sum(data.split.values(), [])

    def wrap_predictor(fn):
        return {r: score_record_predictions(fn(r), data.U[r], data.lab[r], data.samp[r], data.fs) for r in all_rids}

    res["Persistence"] = wrap_predictor(lambda r: predict_persistence(data.U[r]))
    for k in (2, 3, 5, 10):
        res[f"Mean-of-{k}"] = wrap_predictor(lambda r, k=k: predict_meank(data.U[r], k))
    res["Mean-of-10"] = res["Mean-of-10"]

    w_ridge1 = fit_ridge_baseline(data, k=1, val_rids=val_rids)
    res["Ridge, 1 beat"] = wrap_predictor(lambda r: build_ridge_features(data.U[r], 1) @ w_ridge1)

    w_ridge3 = fit_ridge_baseline(data, k=3, val_rids=val_rids)
    res["Ridge, 3 beats"] = wrap_predictor(lambda r: build_ridge_features(data.U[r], 3) @ w_ridge3)

    # 3. Model Optimization or Archived Genomes
    print(f"\n[2/4] Obtaining Reservoir Genomes (Mode: {args.mode})...")
    genomes = {"esn_ga": {}, "dendritic_ga": {}}
    res_seeds = {"ESN + GA": [], "Dendritic + GA": []}
    stored_models = {}

    for model_name, is_dend in [("ESN + GA", False), ("Dendritic + GA", True)]:
        arch_key = "dendritic_ga" if is_dend else "esn_ga"
        for seed in args.seeds:
            if args.mode == "reproduce" and seed in ARCHIVED_GENOMES[arch_key]:
                genome = ARCHIVED_GENOMES[arch_key][seed]
                print(f"  Loaded archived genome for {model_name} (Seed {seed})")
            else:
                print(f"  Running GA search for {model_name} (Seed {seed})...")
                genome, best_val, _ = run_genetic_algorithm(
                    dendritic=is_dend, seed=seed, data=data,
                    population_size=args.pop, generations=args.gens
                )
            genomes[arch_key][seed] = genome

            # Build and score reservoir
            R = build_reservoir(genome, seed=seed, dendritic=is_dend)
            W_out = fit_reservoir_readout(R, data)
            parts = wrap_predictor(lambda r, R=R, Wo=W_out: predict_reservoir(R, Wo, data.U[r]))
            res_seeds[model_name].append(parts)

            if seed == args.seeds[0]:
                stored_models[model_name] = (R, W_out)

    # Average reservoir performance across seeds
    for m in ["ESN + GA", "Dendritic + GA"]:
        averaged = {}
        for r in all_rids:
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

    # 4. Generate Results and Reports
    print("\n[3/4] Computing Benchmark Metrics and Statistical CIs...")
    rng = np.random.default_rng(0)
    boot_indices = [rng.integers(0, len(test_rids), len(test_rids)) for _ in range(args.boot)]

    table6_summary = {}
    models_to_report = [
        "Persistence", "Mean-of-2", "Mean-of-3", "Mean-of-5", "Mean-of-10",
        "Ridge, 1 beat", "Ridge, 3 beats", "ESN + GA", "Dendritic + GA"
    ]

    for m in models_to_report:
        test_parts = [res[m][r] for r in test_rids]
        pooled = compute_pooled_metrics(test_parts)

        boot_rmse = [compute_pooled_metrics([res[m][test_rids[i]] for i in b_idx])["rmse"] for b_idx in boot_indices]
        ci_lo, ci_hi = compute_bootstrap_ci(boot_rmse)
        table6_summary[m] = {
            "rmse": pooled["rmse"],
            "auc": pooled["auc"],
            "ap": pooled["ap"],
            "ci": (ci_lo, ci_hi)
        }

    # Alarm Evaluation (Table 8)
    print("\n[4/4] Evaluating Alarm Rules and Generating Output Artifacts...")
    alarm_summary = []
    for m in ["Persistence", "Mean-of-10", "ESN + GA", "Dendritic + GA"]:
        test_parts = [res[m][r] for r in test_rids]
        calib_scores = np.concatenate([res[m][r]["sc"] for r in calib_rids])
        for mode in ["global", "per_patient"]:
            for q in [0.95, 0.99]:
                alm = evaluate_threshold_alarms(test_parts, mode=mode, q=q, calib_scores=calib_scores)
                alarm_summary.append({
                    "model": m, "mode": mode, "q": q, **alm
                })

    # Save outputs
    out_json = {
        "metadata": {
            "split": args.split,
            "beats": args.beats,
            "mode": args.mode,
            "runtime_seconds": time.time() - t_start
        },
        "table6_forecasting": table6_summary,
        "table8_alarms": alarm_summary,
        "genomes": genomes
    }

    json_path = os.path.join(args.out, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    # Save Markdown Summary
    md_lines = [
        "# Benchmark Results Summary",
        f"**Mode:** {args.mode} | **Beats:** {args.beats} | **Execution Time:** {time.time() - t_start:.1f}s\n",
        "## Table 6: Next-Beat Forecasting & Anomaly Ranking (22 Test Records)",
        "| Model | RMSE [95% CI] | ROC-AUC | Avg. Precision |",
        "| :--- | :--- | :---: | :---: |"
    ]
    for m in models_to_report:
        d = table6_summary[m]
        md_lines.append(f"| {m} | {d['rmse']:.4f} [{d['ci'][0]:.4f}, {d['ci'][1]:.4f}] | {d['auc']:.3f} | {d['ap']:.3f} |")

    md_lines.extend([
        "\n## Table 8: Thresholded Clinical Alarms",
        "| Model | Rule | q | Precision | Recall | F1 | False Alarms / Hour |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
    ])
    for a in alarm_summary:
        md_lines.append(f"| {a['model']} | {a['mode']} | {a['q']} | {a['precision']:.3f} | {a['recall']:.3f} | {a['f1']:.3f} | {a['fa_per_hour']:.1f} |")

    md_path = os.path.join(args.out, "BENCHMARK_REPORT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n[DONE] Results saved to:\n  - JSON: {json_path}\n  - Markdown: {md_path}")
    print(f"Total benchmark elapsed time: {time.time() - t_start:.1f} seconds.")


if __name__ == "__main__":
    main()

