# CAST

**C**ausality-guided flight-parameter prediction **A**cross heterogeneous atmospheric conditions using **S**patio-**T**emporal modeling.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/pytorch-2.1-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

CAST is a causality-guided time-series forecasting framework for reliable flight-parameter prediction under heterogeneous and adverse operating conditions. This release includes the CAST implementation, the reported baselines, experiment configurations, multi-seed evaluation scripts, and lightweight datasets for immediate functional validation.

> **Reproducibility**
>
> - **Quick functional check:** bundled lightweight data + `python quick_validation.py`
> - **Single full-data experiment:** download the full dataset + `python run.py ...`
> - **Tables 1–3:** download the full datasets + `python batch_run.py ...`
> - **Detailed protocol:** [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)

---

## Quick start

### 1. Install

From the repository root:

```bash
python -m pip install --upgrade pip
pip install -e .
```

Verify the installation:

```bash
python -c "import torch; import utils; print('CAST environment OK')"
```

The manuscript experiments were developed with Python 3.8+, PyTorch 2.1, and CUDA 12.2.

### 2. Run the bundled lightweight validation

No additional dataset download is required for the functional check:

```bash
python quick_validation.py
```

The bundled datasets contain reduced numbers of samples and are provided **only to verify that the released code executes correctly**. Their numerical results are therefore not expected to match Tables 1–3.

### 3. Run manuscript-scale experiments

For manuscript-scale evaluation:

1. download the corresponding full dataset from the links below;
2. place it at the specified path, replacing the lightweight file where applicable; and
3. run `run.py` for a single experiment or `batch_run.py` for table-level reproduction.

No modification to the model implementation or released experiment configurations is required.

---

## Method overview

CAST combines three coordinated components:

| Component                                                | Role                                                                                                 | Main implementation                         |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **FPG — Flight-Parameter Graph**                  | Organizes flight-variable types and learns condition-dependent directed interactions.                | `model/layer.py`                          |
| **STDC — Structured Temporal Dependency Capture** | Models temporally ordered dependencies that remain informative across changing operating conditions. | `model/attention.py`, `model/resnet.py` |
| **CDG — Conditional Diffusion Guidance**          | Refines multi-step predictions through conditional denoising to reduce residual error accumulation.  | `model/diffusion_model.py`                |

---

## Data

The repository is designed around two data scales:

| Mode                              | Data                           | Entry point                  | Purpose                                      |
| --------------------------------- | ------------------------------ | ---------------------------- | -------------------------------------------- |
| **Functional validation**   | Bundled lightweight datasets   | `quick_validation.py`      | Verify installation and end-to-end execution |
| **Manuscript reproduction** | Full datasets downloaded below | `run.py`, `batch_run.py` | Reproduce manuscript-scale experiments       |

### Full-data sources

| Dataset                       | Full-data source                                                                                 | Expected path                                                   | Used in |
| ----------------------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- | ------- |
| **ETTh1**               | https://github.com/zhouhaoyi/ETDataset                                                           | `data/ETTh1.csv`                                              | Table 1 |
| **Exchange Rate**       | https://github.com/laiguokun/multivariate-time-series-data/tree/master/exchange_rate             | `data/exchange_rate/exchange_rate.csv`                        | Table 1 |
| **NY Weather**          | https://data.cityofnewyork.us/dataset/Hyperlocal-Temperature-Monitoring/qdq3-9eqn/about_data     | `data/weather/Hyperlocal_Temperature_Monitoring_20240928.csv` | Table 2 |
| **DMSP-Par**            | https://doi.org/10.5281/zenodo.4281122                                                           | `data/pci/DMSP_final_training.csv`                            | Table 2 |
| **C919 — Crosswind**    | https://anonymous.4open.science/r/Flight_Data-609D                                               | `data/C919/1.txt`                                             | Table   |
| **C919 — Gust**         |  https://anonymous.4open.science/r/Flight_Data-609D                                              | `data/C919/2.txt`                                             | Table 3 |
| **C919 — Icing**        |  https://anonymous.4open.science/r/Flight_Data-609D                                              | `data/C919/3.txt`                                             | Table 3 |
| **C919 — Hot updraft**  |  https://anonymous.4open.science/r/Flight_Data-609D                                              | `data/C919/4.txt`                                             | Table 3 |

---

## Reproducing the manuscript results

All reported multi-run experiments use five independent random seeds:

```text
2023, 2024, 2025, 2026, 2027
```

For the exact evaluation and aggregation procedure, see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

### Tables 1 and 2 — public datasets

Reproduce the complete comparisons on ETTh1, Exchange Rate, NY Weather, and DMSP-Par:

```bash
python batch_run.py \
  --datasets etth1,exchange_rate,weather,pci \
  --models cast,tsamd,xpatch,duet,pathformer,pdf,itransformer,fedformer,autoformer \
  --seeds 2023,2024,2025,2026,2027
```

CAST-only:

```bash
python batch_run.py \
  --datasets etth1,exchange_rate,weather,pci \
  --models cast \
  --seeds 2023,2024,2025,2026,2027
```

### Table 3 — C919 real-flight scenarios

Table 3 compares xPatch, Duet, and CAST under four adverse-flight scenarios:

```bash
python batch_run.py \
  --datasets c919_1,c919_2,c919_3,c919_4 \
  --models xpatch,duet,cast \
  --seeds 2023,2024,2025,2026,2027
```

| Dataset key | Scenario    |
| ----------- | ----------- |
| `c919_1`  | Crosswind   |
| `c919_2`  | Gust        |
| `c919_3`  | Icing       |
| `c919_4`  | Hot updraft |

---

## Evaluation and outputs

For each independent seed, the released evaluation pipeline computes:

- **Global metrics:** MAE, MSE, RMSE
- **Tail-focused local metrics:** MaxAE, MaxError, MinError

Test mini-batch predictions produced by the evaluation loader are concatenated before metric computation. RMSE is calculated independently for each run as:

```text
RMSE = sqrt(MSE)
```

The batch runner retains per-seed results and cross-seed summaries:

```text
results/results_raw.json
    Per-model, per-dataset, per-seed metrics.

results/results_aggregated.json
    Cross-seed summary statistics.

results/{model}_{dataset}_seed{seed}.log
    Log for each independent run.
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full evaluation protocol.

---

## Repository structure

```text
cast/
├── run.py                  # Single full-data experiment
├── batch_run.py            # Multi-model / multi-dataset / multi-seed reproduction
├── quick_validation.py     # Lightweight functional validation
├── pyproject.toml
├── configs/                # Manuscript experiment configurations
├── data/                   # Bundled lightweight data / full-data target paths
├── model/                  # CAST implementation
├── models/                 # Baseline implementations
├── layers/                 # Shared neural-network components
├── algorithm/              # Dataset preprocessing and loading
├── engine/                 # Training and evaluation pipeline
├── utils/                  # Configuration, metrics, registry, and logging
├── tests/
├── REPRODUCIBILITY.md
├── LICENSE
└── README.md
```

---

## Supported models

| CLI name                               | Model                  |
| -------------------------------------- | ---------------------- |
| `cast`                               | CAST                   |
| `tsamd`                              | AMD                    |
| `xpatch`                             | xPatch                 |
| `duet`                               | Duet                   |
| `pathformer`                         | Pathformer             |
| `pdf`                                | PDF                    |
| `itransformer`                       | iTransformer           |
| `fedformer`                          | FEDformer              |
| `autoformer`                         | Autoformer             |
| `ogat`, `otem`, `ospa`, `onet` | CAST ablation variants |

---

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.
