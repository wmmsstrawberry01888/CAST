# Reproducibility Guide

This document provides step-by-step instructions for validating the released CAST implementation and reproducing the main experimental results reported in the manuscript.

The repository is organized to support two complementary use cases:

1. **Immediate functional validation** using the lightweight processed datasets bundled with the repository.
2. **Manuscript-scale reproduction** using the corresponding full datasets downloaded from the sources listed below.

The lightweight and full datasets use the same file formats, directory structure, and loading interface. To switch from lightweight validation to full-data experiments, download the full dataset and replace the corresponding lightweight file at the same path. No changes to the source code or experiment configurations are required.

---

## 1. Installation

From the repository root:

```bash
python -m pip install --upgrade pip
pip install -e .
```

Verify that the package and the `utils` module are available:

```bash
python -c "import torch; import utils; print('CAST environment OK')"
```

The manuscript experiments were implemented with:

- **Python:** 3.8+
- **PyTorch:** 2.1
- **CUDA:** 12.2
- **GPU:** NVIDIA RTX A6000

GPU acceleration is recommended for manuscript-scale experiments. The bundled lightweight datasets can be used for a faster functional check.

---

## 2. Data organization

### 2.1 Lightweight datasets bundled with the repository

To make the released code directly runnable, the files under `data/` are lightweight processed versions of the datasets used in the manuscript.

These files are provided for:

- data-loading verification;
- end-to-end training and evaluation checks;
- testing the released experiment pipeline; and
- confirming that result files are generated correctly.

> **Important:** because the bundled files contain fewer samples, their numerical results are **not expected to match Tables 1–3 in the manuscript**.

A lightweight functional validation can be launched directly with:

```bash
python quick_validation.py
```

### 2.2 Full datasets for manuscript-scale reproduction

For full-data experiments, download the corresponding dataset and replace the lightweight file at the path shown below.

| Dataset                       | Full-data source                                                                             | Replace this file                                               | Used in |
| ----------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ------- |
| **ETTh1**               | https://github.com/zhouhaoyi/ETDataset                                                       | `data/ETTh1.csv`                                              | Table 1 |
| **Exchange Rate**       | https://github.com/laiguokun/multivariate-time-series-data/tree/master/exchange_rate         | `data/exchange_rate/exchange_rate.csv`                        | Table 1 |
| **NY Weather**          | https://data.cityofnewyork.us/dataset/Hyperlocal-Temperature-Monitoring/qdq3-9eqn/about_data | `data/weather/Hyperlocal_Temperature_Monitoring_20240928.csv` | Table 2 |
| **DMSP-Par**            | https://doi.org/10.5281/zenodo.4281122                                                       | `data/pci/DMSP_final_training.csv`                            | Table 2 |
| **C919 — Crosswind**    | https://anonymous.4open.science/r/Flight_Data-609D                                           | `data/C919/1.txt`                                             | Table 3 |
| **C919 — Gust**         | https://anonymous.4open.science/r/Flight_Data-609D                                           | `data/C919/2.txt`                                             | Table 3 |
| **C919 — Icing**        | https://anonymous.4open.science/r/Flight_Data-609D                                           | `data/C919/3.txt`                                             | Table 3 |
| **C919 — Hot updraft**  | https://anonymous.4open.science/r/Flight_Data-609D                                           | `data/C919/4.txt`                                             | Table 3 |

After replacement, use `run.py` for an individual full-data experiment or `batch_run.py` for table-level reproduction. The reviewer-facing reproduction commands verified for this release are provided below.

---

## 3. Evaluation protocol

The released reproduction pipeline follows the evaluation procedure described in the manuscript.

### 3.1 Chronological data split

Each dataset is divided chronologically into:

```text
70% training / 20% validation / 10% test
```

The validation set is used for model selection, and final metrics are computed on the held-out test set.

### 3.2 Five independent random seeds

All reported multi-run results use five independent random seeds:

```text
2023, 2024, 2025, 2026, 2027
```

For every seed:

1. the model is initialized and trained independently;
2. the best checkpoint is selected using the validation set;
3. predictions from the test mini-batches produced by the released evaluation loader are concatenated;
4. test metrics are computed from the concatenated test predictions; and
5. the per-seed metrics are saved before cross-seed aggregation.

### 3.3 Metrics

The evaluator reports the six metrics used in the manuscript:

- **MAE**
- **MSE**
- **RMSE**
- **MaxAE**
- **MaxError**
- **MinError**

RMSE is computed independently for each run as:

```text
RMSE = sqrt(MSE)
```

using that run's test-set MSE.

The final reported value for each metric is the arithmetic mean across the five independent seeds. Standard deviations and all seed-level results are retained for independent verification.

---

## 4. Quick functional validation

To verify installation, data loading, model execution, evaluation, and result writing with the bundled lightweight datasets:

```bash
python quick_validation.py
```

This command is intended for **functional verification only**. Because the bundled datasets are reduced versions of the full datasets, the resulting numbers are not expected to match the manuscript tables.

---

## 5. Reproducing Tables 1 and 2

Tables 1 and 2 use the four public datasets: ETTh1, Exchange Rate, NY Weather, and DMSP-Par.

Before running manuscript-scale reproduction, replace the bundled lightweight files with the corresponding full datasets listed in Section 2.2.

### Complete public-dataset comparison

```bash
python batch_run.py \
  --datasets etth1,exchange_rate,weather,pci \
  --models cast,tsamd,xpatch,duet,pathformer,pdf,itransformer,fedformer,autoformer \
  --seeds 2023,2024,2025,2026,2027
```

### CAST-only verification

```bash
python batch_run.py \
  --datasets etth1,exchange_rate,weather,pci \
  --models cast \
  --seeds 2023,2024,2025,2026,2027
```

---

## 6. Reproducing Table 3: C919 real-flight scenarios

Table 3 evaluates xPatch, Duet, and CAST under four representative C919 adverse-flight scenarios.

After downloading the full C919 data and replacing the corresponding lightweight files, reproduce the complete table with:

```bash
python batch_run.py \
  --datasets c919_1,c919_2,c919_3,c919_4 \
  --models xpatch,duet,cast \
  --seeds 2023,2024,2025,2026,2027
```

Scenario mapping:

| Dataset key | Scenario    | File                |
| ----------- | ----------- | ------------------- |
| `c919_1`  | Crosswind   | `data/C919/1.txt` |
| `c919_2`  | Gust        | `data/C919/2.txt` |
| `c919_3`  | Icing       | `data/C919/3.txt` |
| `c919_4`  | Hot updraft | `data/C919/4.txt` |

---

## 7. Output files

The standard batch runner writes:

```text
results/results_raw.json
```

Per-model, per-dataset, per-seed test metrics.

```text
results/results_aggregated.json
```

Cross-seed summary statistics.

```text
results/{model}_{dataset}_seed{seed}.log
```

Detailed log for each independent run.

For each model-dataset pair, the aggregated output should contain all six manuscript metrics:

```text
MAE, MSE, RMSE, MaxAE, MaxError, MinError
```

---

## 8. Reproducibility checklist

Before comparing reproduced values with the manuscript, confirm that:

- [ ] the full dataset has replaced the bundled lightweight file;
- [ ] the released YAML configuration is being used;
- [ ] the chronological split is 70% / 20% / 10%;
- [ ] the experiment uses the exact five seeds reported above;
- [ ] the best checkpoint is selected using the validation set;
- [ ] evaluation is performed on the held-out test set;
- [ ] predictions from the test mini-batches produced by the released evaluation loader are concatenated before metric computation;
- [ ] RMSE is computed independently for each seed as `sqrt(MSE)`;
- [ ] all six metrics are saved for each seed; and
- [ ] final table values are obtained by averaging the five seed-level metrics.

---

## 9. Troubleshooting

### `ImportError: No module named 'utils'`

Make sure the repository has been installed from its root:

```bash
pip install -e .
python -c "import utils; print('utils package loaded')"
```

The released repository includes `utils/__init__.py`.

### Dataset not found

Check whether the dataset file is located at the exact path shown in Section 2.2.

For lightweight functional validation, use the bundled files.

For manuscript-scale reproduction, replace the bundled file with the corresponding full dataset **at the same path**.

### Out-of-memory errors

Manuscript reproduction should use the released YAML configurations. If hardware limitations require modifying the batch size, the resulting experiment no longer corresponds exactly to the configuration used for the reported manuscript values.

### Dependency problems

Reinstall from the repository root:

```bash
pip install -e .
```

---

## 10. Notes for reviewers

The lightweight datasets bundled with the repository are intended to make the code immediately executable and to verify the released software workflow without requiring large downloads.

For independent numerical verification of the manuscript tables:

1. download the corresponding full dataset;
2. replace the bundled lightweight file at the same path;
3. run the released table command with the five reported seeds; and
4. inspect both `results_raw.json` and `results_aggregated.json`.

No changes to the model implementation, data-loading code, or experiment configurations are required when switching from the bundled lightweight data to the full datasets.
