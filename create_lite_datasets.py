"""
Create lightweight datasets for quick reviewer validation.
Extracts first 1000 samples from each dataset for fast testing.
"""
import pandas as pd
import os
from pathlib import Path

def create_lite_dataset(source_path, target_path, n_samples=1000,
                        sep=",", encoding=None):
    """Extract first n_samples from a dataset.

    ``sep`` / ``encoding`` let us handle non-standard sources such as the C919
    flight logs (tab-separated, gb2312-encoded). The lite copy is written back
    in the SAME format so the registered preprocessing module reads it the same
    way as the full file.
    """
    print(f"Processing: {source_path}")

    if not os.path.exists(source_path):
        print(f"  Warning: {source_path} not found, skipping.")
        return False

    # Read the dataset
    df = pd.read_csv(source_path, sep=sep, encoding=encoding)
    original_len = len(df)

    # Take first n_samples (including header already read)
    df_lite = df.head(n_samples)

    # Create target directory if needed
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # Save lite version in the same format as the source
    df_lite.to_csv(target_path, index=False, sep=sep, encoding=encoding)

    print(f"  [OK] Created: {target_path}")
    print(f"    Original: {original_len} samples -> Lite: {len(df_lite)} samples")
    return True

def main():
    """Create lite versions of all benchmark datasets."""

    base_dir = Path(__file__).parent / "data"
    lite_dir = base_dir / "lite"
    n_samples = 1000  # Small enough for quick testing

    # Dataset mappings: source -> (lite target, read/write options)
    # C919 flight logs are tab-separated and gb2312-encoded, so they need
    # explicit sep/encoding; the CSV benchmarks use pandas defaults.
    datasets = {
        "ETTh1.csv": ("lite/ETTh1.csv", {}),
        "exchange_rate/exchange_rate.csv": ("lite/exchange_rate.csv", {}),
        "weather/Hyperlocal_Temperature_Monitoring_20240928.csv": ("lite/weather.csv", {}),
        "pci/pci.csv": ("lite/pci.csv", {}),
        "C919/1.txt": ("lite/C919_1.txt", {"sep": "\t", "encoding": "gb2312"}),
    }

    print("=" * 70)
    print("Creating Lightweight Datasets for Reviewer Validation")
    print("=" * 70)
    print(f"Target: {n_samples} samples per dataset")
    print()

    success_count = 0

    for source, (target, opts) in datasets.items():
        source_path = base_dir / source
        target_path = base_dir / target

        if create_lite_dataset(str(source_path), str(target_path), n_samples, **opts):
            success_count += 1
        print()

    print("=" * 70)
    print(f"Summary: Created {success_count}/{len(datasets)} lite datasets")
    print(f"Location: {lite_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
