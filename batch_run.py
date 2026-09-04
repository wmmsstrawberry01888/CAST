"""
Batch runner: runs all models on all available datasets and collects results.
Supports multi-seed averaging for robust evaluation.

Usage:
    # Single-seed run (fast, for debugging)
    python batch_run.py

    # Multi-seed averaging (recommended for paper results)
    python batch_run.py --seeds 2023,2024,2025,2026,2027

    # Subset of models/datasets
    python batch_run.py --models cast,autoformer --datasets etth1,weather
"""

import subprocess
import sys
import os
import re
import json
import datetime
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from utils.dataset_names import display_name

# All models to evaluate
ALL_MODELS = [
    "cast",
    "ogat",
    "otem",
    "ospa",
    "onet",
    "autoformer",
    "fedformer",
    "pdf",
    "xpatch",
    "itransformer",
    "pathformer",
    "tsamd",
    "duet",
]

# Datasets registered for manuscript-scale reproduction.
# Full datasets are downloaded separately as described in README.md.
DATASETS = {
    # Table 1: Public benchmarks
    "etth1": "configs/table1_etth1.yaml",
    "exchange_rate": "configs/table1_exchange.yaml",

    # Table 2: Public benchmarks
    "weather": "configs/table2_weather.yaml",
    "pci": "configs/table2_pci.yaml",

    # Table 3: C919 flight parameter data (all 4 scenarios included)
    "c919_1": "configs/table3_c919_1.yaml",
    "c919_2": "configs/table3_c919_2.yaml",
    "c919_3": "configs/table3_c919_3.yaml",
    "c919_4": "configs/table3_c919_4.yaml",
}

# Dataset metadata for reporting
DATASET_METADATA = {
    "etth1": {"table": "Table 1", "desc": "ETTh1 - Electricity Transformer Temperature"},
    "exchange_rate": {"table": "Table 1", "desc": "Exchange Rate - 8 currencies"},
    "weather": {"table": "Table 2", "desc": "NYC Weather - Hyperlocal temperature"},
    "pci": {"table": "Table 2", "desc": "DMSP-Par - Plasma density from DMSP satellites"},
    "c919_1": {"table": "Table 3", "desc": "C919 Scenario-1 flight dynamics"},
    "c919_2": {"table": "Table 3", "desc": "C919 Scenario-2 flight dynamics"},
    "c919_3": {"table": "Table 3", "desc": "C919 Scenario-3 flight dynamics"},
    "c919_4": {"table": "Table 3", "desc": "C919 Scenario-4 flight dynamics"},
}

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment(model: str, data: str, config_path: str, seed: int) -> dict:
    """Run one model-dataset experiment with a given seed and return extracted metrics."""
    tag = f"{model}_{data}_seed{seed}"
    log_file = RESULTS_DIR / f"{tag}.log"

    print(f"\n{'='*70}")
    print(f"Running: {model} on {display_name(data)} (seed={seed})  ->  {log_file}")
    print(f"{'='*70}")

    cmd = [
        sys.executable, "run.py",
        "--config", config_path,
        "--model", model,
        "--data", data,
        "--seed", str(seed),
    ]

    with open(log_file, "w", encoding="utf-8") as f:
        # Redirect subprocess output to avoid interfering with progress bar
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                cwd=os.path.dirname(os.path.abspath(__file__)))

    # Extract held-out test-set metrics from log
    metrics = _parse_log(log_file)
    metrics["model"] = model
    metrics["data"] = data
    metrics["seed"] = seed
    metrics["returncode"] = result.returncode

    # Print quick summary
    if metrics.get("best_epoch") is not None:
        mae_str = f"{metrics['best_mae']:.4f}" if metrics.get('best_mae') is not None else 'N/A'
        rmse_str = f"{metrics['best_rmse']:.4f}" if metrics.get('best_rmse') is not None else 'N/A'
        mse_str = f"{metrics['best_mse']:.4f}" if metrics.get('best_mse') is not None else 'N/A'
        print(f"  [OK] Done. Best epoch: {metrics['best_epoch']}, "
              f"MAE: {mae_str}, RMSE: {rmse_str}, MSE: {mse_str}")
    else:
        print(f"  [FAIL] Failed or no metrics found (code={result.returncode})")

    return metrics


def _parse_log(log_path: Path) -> dict:
    """Parse the output log to extract the held-out TEST-set metrics.

    run.py emits a single machine-parseable line after independent test-set
    evaluation::

        TEST_METRICS_JSON: {"test_mae": ..., "test_rmse": ..., ...}

    We parse that line rather than scraping the trainer's tqdm progress bar.
    The progress bar reports *training-set* metrics of the final epoch, which
    are not a valid basis for the reported tables; the JSON line carries the
    metrics computed once on the strictly held-out test set. The summary keys
    (best_mae / best_mse / best_rmse) are populated from the test metrics so
    the downstream aggregation averages held-out test scores.
    """
    metrics = {}
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except FileNotFoundError:
        return metrics

    # Extract the last TEST_METRICS_JSON line (a run emits exactly one; k-fold
    # emits one per fold, in which case the last fold's line is kept here — the
    # per-fold JSON is still available in the raw log for auditing).
    test = None
    for m in re.finditer(r'TEST_METRICS_JSON:\s*(\{.*\})', content):
        try:
            test = json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            continue

    if test is not None:
        # Map held-out test metrics onto the summary keys used by the table.
        if test.get("test_mae") is not None:
            metrics["best_mae"] = test["test_mae"]
        if test.get("test_mse") is not None:
            metrics["best_mse"] = test["test_mse"]
        if test.get("test_rmse") is not None:
            metrics["best_rmse"] = test["test_rmse"]
        if test.get("test_max_ae") is not None:
            metrics["best_max_ae"] = test["test_max_ae"]
        if test.get("test_max_error") is not None:
            metrics["best_max_error"] = test["test_max_error"]
        if test.get("test_min_error") is not None:
            metrics["best_min_error"] = test["test_min_error"]
        if test.get("best_val_loss") is not None:
            metrics["best_loss"] = test["best_val_loss"]
            metrics["best_val_loss"] = test["best_val_loss"]
        if test.get("best_epoch") is not None:
            metrics["best_epoch"] = test["best_epoch"]
        # Keep the full held-out test metric set for auditing / richer tables.
        metrics["test_metrics"] = {k: v for k, v in test.items()
                                   if k.startswith("test_")}
        return metrics

    # Fallback: no JSON line found (e.g. a run that crashed before evaluation).
    # Recover best-epoch info if present so the row is not silently blank.
    for line in content.split("\n"):
        if "Best epoch:" in line:
            parts = line.split(",")
            for p in parts:
                p = p.strip()
                if p.startswith("Best epoch:"):
                    try:
                        metrics["best_epoch"] = int(p.split(":")[1].strip())
                    except (ValueError, IndexError):
                        pass
                if p.startswith("Best val loss:"):
                    try:
                        metrics["best_val_loss"] = float(p.split(":")[1].strip())
                    except (ValueError, IndexError):
                        pass

    return metrics


def _aggregate_seeds(all_results: list, seeds: list) -> list:
    """Aggregate metrics across seeds for each model-dataset pair."""
    from collections import defaultdict

    # Group by (model, data)
    grouped = defaultdict(list)
    for r in all_results:
        if r.get('returncode') == 0:
            key = (r['model'], r['data'])
            grouped[key].append(r)

    aggregated = []
    for (model, data), runs in grouped.items():
        if len(runs) < len(seeds):
            print(f"WARNING: {model} on {display_name(data)} completed only {len(runs)}/{len(seeds)} seeds")

        # Compute mean ± std for each metric
        metrics = {}
        for metric_key in ['best_mae', 'best_mse', 'best_rmse',
                            'best_max_ae', 'best_max_error', 'best_min_error',
                            'best_loss']:
            values = [r[metric_key] for r in runs if metric_key in r and r[metric_key] is not None]
            if values:
                metrics[metric_key] = np.mean(values)
                metrics[f"{metric_key}_std"] = np.std(values, ddof=1) if len(values) > 1 else 0.0

        aggregated.append({
            'model': model,
            'data': data,
            'n_seeds': len(runs),
            **metrics,
        })

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation with multi-seed averaging")
    parser.add_argument("--seeds", type=str, default="2025",
                        help="Comma-separated seed values (e.g., '2023,2024,2025,2026,2027')")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model subset (default: all)")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset subset (default: all)")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    models = [m.strip() for m in args.models.split(",")] if args.models else ALL_MODELS
    dataset_keys = [d.strip() for d in args.datasets.split(",")] if args.datasets else list(DATASETS.keys())
    datasets = {k: DATASETS[k] for k in dataset_keys if k in DATASETS}

    all_results = []
    start_time = datetime.datetime.now()
    total_experiments = len(models) * len(datasets) * len(seeds)

    print(f"\n{'='*80}")
    print(f"Batch run started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(f"Models:   {models}")
    print(f"Datasets: {[display_name(k) for k in datasets.keys()]}")
    print(f"Seeds:    {seeds}")
    print(f"Total experiments: {total_experiments}")
    print(f"{'='*80}\n")

    # Create progress bar for all experiments
    with tqdm(total=total_experiments, desc="Overall Progress",
              unit="exp", ncols=100, colour='green', position=0) as pbar:
        for data_name, config_path in datasets.items():
            for model in models:
                for seed in seeds:
                    exp_start = datetime.datetime.now()

                    # Update progress bar description with current experiment
                    pbar.set_description(f"Running {model} on {display_name(data_name)} (seed={seed})")
                    pbar.refresh()

                    metrics = run_experiment(model, data_name, config_path, seed)
                    metrics["duration_sec"] = (datetime.datetime.now() - exp_start).total_seconds()
                    all_results.append(metrics)

                    # Save intermediate results
                    with open(RESULTS_DIR / "results_raw.json", "w") as f:
                        json.dump(all_results, f, indent=2)

                    # Update progress bar with metrics
                    success_count = sum(1 for r in all_results if r.get('returncode') == 0)
                    pbar.set_postfix({
                        'success': f"{success_count}/{len(all_results)}",
                        'MAE': f"{metrics.get('best_mae', 0):.4f}" if metrics.get('best_mae') else 'N/A'
                    })
                    pbar.update(1)
                    pbar.refresh()

    # Aggregate across seeds
    aggregated = _aggregate_seeds(all_results, seeds)
    with open(RESULTS_DIR / "results_aggregated.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    # Final summary
    end_time = datetime.datetime.now()
    total_duration = end_time - start_time

    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY (held-out test set, averaged across seeds)")
    print("="*80)
    print(f"Total time: {total_duration}")
    print(f"Successful: {sum(1 for r in all_results if r.get('returncode') == 0)}/{len(all_results)}")
    print()

    # Print aggregated table
    header = f"{'Model':<16} {'Data':<18} {'MAE':>12} {'MSE':>12} {'RMSE':>12} {'Seeds':>6}"
    print(header)
    print("-" * len(header))
    for r in aggregated:
        mae = r.get('best_mae')
        mae_std = r.get('best_mae_std', 0.0)
        mse = r.get('best_mse')
        mse_std = r.get('best_mse_std', 0.0)
        rmse = r.get('best_rmse')
        rmse_std = r.get('best_rmse_std', 0.0)
        if mae is not None:
            # All of MAE/MSE/RMSE come from the held-out test metrics; format
            # each defensively so a missing value doesn't crash the summary.
            def _fmt(val, std):
                return f"{val:>6.4f}±{std:>4.4f}" if val is not None else f"{'N/A':>12}"
            print(f"{r['model']:<16} {display_name(r['data']):<18} "
                  f"{_fmt(mae, mae_std)} "
                  f"{_fmt(mse, mse_std)} "
                  f"{_fmt(rmse, rmse_std)} "
                  f"{r['n_seeds']:>6}")
        else:
            print(f"{r['model']:<16} {display_name(r['data']):<18} {'FAILED':>12} {'FAILED':>12} {'FAILED':>12} {r['n_seeds']:>6}")

    print(f"\nRaw results: {RESULTS_DIR / 'results_raw.json'}")
    print(f"Aggregated results: {RESULTS_DIR / 'results_aggregated.json'}")

if __name__ == "__main__":
    main()
