#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick Validation Script for Reviewers

This script runs a lightweight version of CAST model training on 5 benchmark datasets
to demonstrate code functionality. It uses small datasets and minimal epochs for fast verification.

Usage:
    python quick_validation.py

Expected runtime: ~5-10 minutes
"""

import os
import re
import sys
import time
import json
import yaml
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

from utils.dataset_names import display_name


def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def create_lite_config(dataset_name, data_path, output_config):
    """Create a lite configuration file for quick validation."""

    # Map dataset names to their lite versions
    lite_name = f"{dataset_name}_lite"

    config = {
        'experiment': {
            'name': f'quick_val_{dataset_name}',
            'data': lite_name,  # Use the registered lite dataset
            'mode': 'standard',
            'seed': 2025,
            'device': 'cuda',
            'k_folds': 1,
            'output_dir': './results/quick_validation',
            'notes': f'Quick validation on {dataset_name} lite dataset'
        },
        'model': {
            'T_h': 12,
            'T_p': 12
        },
        'train': {
            'epochs': 3,
            'lr': 1e-4,
            'batch_size': 16,
            'weight_decay': 1e-5,
            'optimizer': 'adam',
            'early_stop_patience': 5,
            'seed': 2025,
            'lambda_cdg': 1.0,
            'use_amp': False,
            'grad_clip': 0.0,
            'lr_scheduler': 'none'
        },
        'diffusion': {
            'N': 100,
            'sample_steps': 100,
            'epsilon_theta': 'UGnet',
            'beta_end': 0.02,
            'beta_schedule': 'quad',
            'sample_strategy': 'ddpm'
        },
        'data': {
            'name': lite_name,
            'num_features': 3,
            'num_vertices': 4,
            'points_per_hour': 12
        }
    }

    # Dataset-specific settings
    if dataset_name == 'pci':
        config['data']['num_vertices'] = 9
        config['data']['num_features'] = 10
    elif dataset_name == 'c919':
        # C919 flight graph has 5 nodes (time / GPS / sideslip / load / rate).
        config['data']['num_vertices'] = 5
        config['data']['num_features'] = 3
        # C919 raw features span very large ranges (altitude ~0-7865,
        # sideslip +-175); without gradient clipping the CAST + CDG objective
        # diverges to nan. Clip to keep the quick run stable.
        config['train']['grad_clip'] = 1.0

    os.makedirs(os.path.dirname(output_config), exist_ok=True)

    with open(output_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return output_config


def run_experiment(dataset_name, data_path, result_dir):
    """Run a single experiment and return results."""
    start_time = time.time()

    # Create temporary config file
    temp_config = result_dir / f'config_{dataset_name}.yaml'
    create_lite_config(dataset_name, data_path, temp_config)

    # Prepare command
    cmd = [
        sys.executable,
        'run.py',
        '--config', str(temp_config),
        '--model', 'cast'
    ]

    # Run experiment
    log_file = result_dir / f'{dataset_name}.log'
    print(f"Running: {display_name(dataset_name)}")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        with open(log_file, 'w') as log:
            process = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                cwd=os.getcwd()
            )

        elapsed = time.time() - start_time

        # Parse results from log
        mae, rmse = parse_results(log_file)

        mae_disp = f"{mae:.4f}" if mae is not None else "N/A (not found in log)"
        rmse_disp = f"{rmse:.4f}" if rmse is not None else "N/A (not found in log)"

        print(f"[OK] {display_name(dataset_name)} completed in {elapsed:.1f}s")
        print(f"     MAE: {mae_disp}, RMSE: {rmse_disp}")
        if mae is None or rmse is None:
            print(f"     [WARN] Could not parse metrics from {log_file}")
        print()

        return {
            'status': 'success',
            'time': elapsed,
            'mae': mae,
            'rmse': rmse
        }

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"[X] {display_name(dataset_name)} failed after {elapsed:.1f}s")
        print(f"    See log: {log_file}")
        print()
        return {
            'status': 'failed',
            'time': elapsed,
            'mae': None,
            'rmse': None,
            'error': str(e)
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[X] Error: {str(e)}")
        return {
            'status': 'error',
            'time': elapsed,
            'mae': None,
            'rmse': None,
            'error': str(e)
        }


def parse_results(log_file):
    """Parse MAE and RMSE from log file.

    Metrics are emitted in the training progress bar as ``mae=<value>`` and
    ``rmse=<value>`` (e.g. ``mae=1.3830, rmse=2.4016``). We scan the whole log
    and keep the LAST occurrence, which corresponds to the final epoch. The
    values may be wrapped in ANSI color codes, so we match on the numeric token
    directly rather than splitting on whitespace.

    Returns (mae, rmse); either may be None if the metric was never found. We do
    NOT substitute placeholder values — a missing metric means the run failed to
    report one, and masking that with fake numbers is exactly what made four
    different datasets appear to produce identical results.
    """
    mae, rmse = None, None

    mae_re = re.compile(r'mae\s*[=:]\s*([-+]?\d+(?:\.\d+)?)', re.IGNORECASE)
    rmse_re = re.compile(r'rmse\s*[=:]\s*([-+]?\d+(?:\.\d+)?)', re.IGNORECASE)

    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        mae_matches = mae_re.findall(content)
        rmse_matches = rmse_re.findall(content)

        if mae_matches:
            mae = float(mae_matches[-1])
        if rmse_matches:
            rmse = float(rmse_matches[-1])
    except Exception as e:
        print(f"Warning: Could not parse results from {log_file}: {e}")

    return mae, rmse


def print_summary(results, total_time, result_dir):
    """Print and save summary of all experiments."""
    print_header("Quick Validation Summary")

    successful = sum(1 for r in results.values() if r['status'] == 'success')
    total = len(results)

    print(f"Status: {successful}/{total} experiments completed successfully")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)\n")

    print("Results by dataset:")
    print(f"{'Dataset':<20} {'Status':<10} {'Time':<10} {'MAE':<12} {'RMSE':<12}")
    print("-" * 70)

    for dataset, result in results.items():
        status = "[OK]" if result['status'] == 'success' else "[X] Failed"
        time_str = f"{result['time']:.1f}s"
        mae_str = f"{result['mae']:.4f}" if result['mae'] is not None else "N/A"
        rmse_str = f"{result['rmse']:.4f}" if result['rmse'] is not None else "N/A"

        print(f"{display_name(dataset):<20} {status:<10} {time_str:<10} {mae_str:<12} {rmse_str:<12}")

    print()
    print(f"Log files saved in: {result_dir}/")

    # Save JSON summary
    summary_file = result_dir / 'summary.json'
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'total_experiments': total,
        'successful': successful,
        'total_time_seconds': total_time,
        'results': results
    }

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"Summary saved to: {summary_file}")

    if successful < total:
        print("\n[WARNING] Some experiments failed. Please check the log files.")
        return 1
    else:
        print("\n[SUCCESS] All experiments completed successfully!")
        print("The code is ready for reviewer validation.")
        return 0


def main():
    """Main validation workflow."""
    print_header("CAST Model - Quick Validation for Reviewers")

    print("This script will:")
    print("  1. Create lightweight datasets (1000 samples each)")
    print("  2. Train CAST model on 5 benchmarks (3 epochs each)")
    print("  3. Generate validation summary")
    print()
    print("Expected time: ~5-10 minutes")
    print()

    # Step 1: Create lite datasets (skip if already exists)
    print_header("Step 1: Creating Lightweight Datasets")
    try:
        subprocess.run([sys.executable, "create_lite_datasets.py"], check=True)
    except subprocess.CalledProcessError:
        print("[X] Failed to create lite datasets")
        return 1

    # Step 2: Run experiments
    print_header("Step 2: Running Quick Validation Experiments")

    # Prepare result directory
    result_dir = Path('./results/quick_validation')
    result_dir.mkdir(parents=True, exist_ok=True)

    # Define experiments
    datasets = [
        ('etth1', 'ETTh1.csv'),
        ('exchange_rate', 'exchange_rate.csv'),
        ('weather', 'weather.csv'),
        ('pci', 'pci.csv'),
        ('c919', 'C919_1.txt')
    ]

    # Run all experiments
    total_start = time.time()
    results = {}

    for dataset_name, data_path in datasets:
        print("\n" + "=" * 70)
        print(f"Running: {display_name(dataset_name)}")
        print("=" * 70)
        results[dataset_name] = run_experiment(dataset_name, data_path, result_dir)

    total_time = time.time() - total_start

    # Step 3: Print summary
    return print_summary(results, total_time, result_dir)


if __name__ == '__main__':
    sys.exit(main())
