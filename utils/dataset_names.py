"""
Human-readable dataset labels for console output.

The internal dataset keys (e.g. 'pci', 'c919_1') drive config logic, registry
lookups, and file paths and must NOT change. This module maps those keys to the
paper-facing names shown to users, so reviewers see the dataset names used in
the paper (e.g. 'DMSP-Par' instead of 'pci').
"""

# Canonical display names, keyed by internal dataset name.
DISPLAY_NAMES = {
    'etth1': 'ETTh1',
    'exchange_rate': 'Exchange Rate',
    'weather': 'Weather',
    'pci': 'DMSP-Par',
    # C919 flight-test scenarios
    'c919': 'C919',
    'c919_1': 'C919 Scenario-1',
    'c919_2': 'C919 Scenario-2',
    'c919_3': 'C919 Scenario-3',
    'c919_4': 'C919 Scenario-4',
    'c919_lite': 'C919 (lite)',
    # Lite variants of the public benchmarks
    'etth1_lite': 'ETTh1 (lite)',
    'exchange_rate_lite': 'Exchange Rate (lite)',
    'weather_lite': 'Weather (lite)',
    'pci_lite': 'DMSP-Par (lite)',
}


def display_name(dataset_name):
    """Return the paper-facing label for a dataset key (for printing only).

    Unknown keys are returned unchanged so nothing ever crashes on a new
    dataset; add it to DISPLAY_NAMES to give it a friendly label.
    """
    return DISPLAY_NAMES.get(dataset_name, dataset_name)
