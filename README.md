# Imbalanced_Dataset_Characterization

![Overview](Example_visualization.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)


## ABSTRACT

Despite extensive work on imbalanced classification, the reasons why certain datasets prove more challenging than others remain poorly understood, owing to the intricate interplay of class imbalance with data irregularities such as class overlap, the presence of small disjuncts or noise. We address this gap by introducing the first end-to-end explainable framework that forecasts a dataset’s classification difficulty while simultaneously identifying and quantifying the key data irregularities dictating such assessment. Our contributions are threefold: (1) a suite of GPU-accelerated complexity metrics tailored for imbalanced domains, which act as meta-feature in the proposed framework; (2) two novel, scalable methodologies for computing complexity measures in multiclass settings; and (3) two explainable profiling models, one based on Explainable Boosting Machines (EBMs) and another on GAMformer, a state-of-the-art additive in-context learning model. On a benchmark of approximately 600 real-world binary and 120 multiclass datasets, our EBMs and GAMformer achieved a $R^2$ of 0.887 and 0.888 for binary classification problems and 0.934 and 0.902 for multiclass settings, respectively, while producing transparent explanations that align with both theoretical expectations and t-SNE visualizations. Finally, we showcase real-world applicability by accurately identifying when SMOTE and cost-sensitive Random Forests, two widely adopted technique for addressing imbalanced domains, will improve classification outcomes across varying data profiles. The proposed framework is computationally efficient (GAMformer requires no training), highly accurate, and readily interpretable, offering a powerful tool for dataset profiling, benchmarking, method development, and meta-learning in imbalanced classification.


**Key Contributions:**

This repository provides a meta-learning framework to **quantify dataset difficulty** and **understand which data-complexity factors drive that difficulty**, using a GAMformer/EBM meta-model trained on a large meta-dataset of classification problems. The main user-facing interface is `domain_analyzer.py`, which lets you analyze *new* datasets in a few lines of code. 

The framework is composed of three main layers:

1. **Meta-dataset extraction** – compute classical complexity measures (`F2`, `F3`, `F4`, `L1`, `L2`, `L3`, `N2`, `N3`, `N4`, `R_aug`, `T1`, `BI³`) and classification-difficulty (CD) for many datasets. 
2. **Meta-model training & runtime benchmarking** – fit GAMformer and EBM regressors to predict CD from complexity features, and benchmark their runtime. 
3. **Domain-level analysis** – `domain_analyzer.py` (GAMformerAnalysis) loads the trained meta-model and uses it to analyze *your* dataset, producing:

   * A **scalar CD prediction** (“how hard is this dataset?”)
   * **Feature-level contributions** for each complexity metric
   * **Class-wise complexity heatmaps** (multiclass)
   * **2D visualizations** (raw or t-SNE) of the data distribution 

---

## Installation

This project assumes a **GPU-enabled** Python environment with CUDA 12.x for RAPIDS/cuML and CuPy.

1. Create and activate an environment (example):

   ```bash
   conda create -n gamformer python=3.11
   conda activate gamformer
   ```

2. Install the core dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file includes the main libraries:

   * `cuml`, `cupy_cuda12x` – GPU-accelerated ML and array operations
   * `numpy`, `pandas`, `scipy`, `scikit_learn`, `seaborn`, `matplotlib`
   * `interpret`, `interpret_core` – Explainable Boosting Machine (EBM)
   * `torch`, `xgboost` (GPU-enabled) 

3. Install additional meta-model dependency via: https://github.com/microsoft/ticl/tree/main/ticl

   `domain_analyzer.py` and `model_performance_Multiclass_analysis.py` both rely on `ticl.prediction.GAMformerRegressor` and `ticl.utils.fetch_model` to load the GAMformer checkpoint.

4. Ensure your GPU drivers and CUDA runtime are compatible with the `cuml` and `cupy_cuda12x` versions in `requirements.txt`.

---

## Repository Structure (Relevant Files)

* `domain_analyzer.py` – **Main entry point for analyzing your own datasets** using a trained GAMformer meta-model. Defines the `GAMformerAnalysis` class and plotting utilities. 
* `meta_dataset_extraction.py` – Builds the **meta-dataset**: computes complexity metrics and CD across many multiclass datasets, using different decomposition strategies (OVO/OVA/closest). 
* `model_performance_Multiclass_analysis.py` – Trains GAMformer and EBM regressors on the meta-dataset, evaluates them via cross-validation, and measures per-fold runtime. 
* `correlation_runtime_analysis.py` – Compares approximation variants of the complexity measures against the exact version in terms of **Pearson correlation with CD** and **median runtime**, outputting a LaTeX table. 
* `scr`contains the GPU-based implementation of the complexity metrics.
* `requirements.txt` – Python dependencies. 
* `LICENSE` – License for this repository (see file for details).

The meta-data used by `domain_analyzer.py` and the regression scripts is expected in a `complexity_data/` directory, typically containing:

* `complexity_n_OVA.npy` – meta-features obtained via OVA (complexity measures)
* `complexity_OVA1.x.npy`- meta-features obtained via closest-OVA with different g(C) functions
* `CD.npy` – corresponding CD scores per dataset

These are produced by running `meta_dataset_extraction.py`. 

---

## Analyzing Datasets with `domain_analyzer.py`

The central abstraction is `GAMformerAnalysis`, which encapsulates:

1. Loading and preprocessing the **meta-dataset** (`complexity_n_OVA.npy`, `CD.npy`) using `preprocess_complexity_data`. 
2. Fetching a **GAMformerRegressor** checkpoint via `ticl.utils.fetch_model`. 
3. Computing **complexity measures** for your dataset using the GPU-accelerated functions from `src` and OVA decomposition strategies from `meta_dataset_extraction.py`.
4. Generating **visual and numeric explanations** through `plot_and_analyze`.

### 1. Requirements for Your Dataset

To analyze a dataset, you need:

* Feature matrix `X`: shape `(n_samples, n_features)` (NumPy array)
* Target vector `y`: shape `(n_samples,)` with integer-encoded class labels
* A `dataset_id`: integer identifier used only for labeling plots

Binary and multiclass datasets are both supported:

* **Binary**: complexity metrics are summarized into a single vector.
* **Multiclass**: complexity metrics are computed in a One-vs-All fashion per class and then aggregated (e.g., mean per metric across classes) for the meta-model.

### 2. Initializing the Analysis Object

In Python:

```python
from pathlib import Path
from domain_analyzer import GAMformerAnalysis

results_dir = Path("complexity_data")

analysis = GAMformerAnalysis(
    complexity_file=results_dir / "complexity_n_OVA.npy",
    cd_file=results_dir / "CD.npy",
    model_str="baam_Daverage_l1e-05_maxnumclasses0_nsamples500_numfeatures10_yencoderlinear_05_08_2024_03_04_01_epoch_40.cpkt",
)
```

When constructing `GAMformerAnalysis`, the following happens automatically:

* `preprocess_complexity_data` aggregates meta-dataset complexity scores and computes final CD (1 − mean CD per dataset).
* `fetch_model(model_str)` resolves and downloads/locates the GAMformer checkpoint.
* A `GAMformerRegressor` is instantiated on `cuda` (if available) or `cpu` and immediately fitted to the meta-features and CD targets.

### 3. One-Shot Dataset Analysis (`plot_and_analyze`)

The easiest way to analyze a dataset is via:

```python
X, y = ...  # load or construct your dataset
dataset_id = 42

cd_pred = analysis.plot_and_analyze(
    full_dataset=(X, y, dataset_id),
    save_fig=True,
    output_dir="analysis_plots",
)
```

This will: 

1. **Compute complexity metrics** using GPU-accelerated functions such as
   `compute_F2_imbalanced_gpu`, `compute_Lx_imbalanced_gpu`, `compute_Nx_imbalanced_gpu`, `hypersphere_T1`, `compute_Raug_imbalanced_gpu`, and `compute_bayes_imbalance_ratio`, combined via OVA decomposition (`one_vs_all_decomposition`).
2. Aggregate metrics into a single feature vector per dataset and feed it into the trained GAMformer meta-model.
3. Obtain:

   * `cd_pred`: predicted **classification difficulty** for the dataset.
   * Additive **feature contributions** for each complexity metric.
4. Produce a multi-panel figure saved to `output_dir`:

   * **Panel 1 (left)** – Horizontal bar plot of metric contributions (local explanation of CD prediction).
   * **Panel 2 (center, multiclass only)** – Class-wise complexity heatmap ordered by importance.
   * **Panel 3 (right)** – 2D data visualization (t-SNE when `n_features > 2`, otherwise raw 2D scatter).

Set `save_fig=False` if you only want the plot on screen and not saved to disk.

### 4. Low-Level Access: Complexity Metrics & Meta-Model Output

If you need direct access to intermediate representations:

```python
# 1) Compute complexity metrics (per metric × per class)
classwise_matrix = analysis.compute_complexity_metrics(X, y)

# 2) Aggregate for the meta-model
if len(np.unique(y)) > 2:  # multiclass
    complexity_for_feat = classwise_matrix.mean(axis=1, keepdims=True).T
else:
    complexity_for_feat = classwise_matrix  # already 1 × n_features

# 3) Get meta-model outputs (prediction + contributions)
outputs = analysis.get_meta_output(complexity_for_feat)

y_pred = outputs["y_pred"]               # shape (1,)
components = outputs["components"]       # additive contributions
sorted_features = outputs["sorted_features"]
sorted_components = outputs["sorted_components"]
```

This is useful if you want to integrate the CD prediction and explanations into custom dashboards or downstream pipelines. 

---

## Meta-Dataset Extraction & Regression Experiments (Context)

Although most users only need `domain_analyzer.py`, the following scripts explain how the meta-model and meta-dataset are obtained.

### `meta_dataset_extraction.py`

* Loads multiple multiclass datasets (`X_raw`, `y_raw`) from `.npy` files.
* Preprocesses them using a scikit-learn pipeline that:

  * Encodes categorical features with `OrdinalEncoder`,
  * Imputes missing values with `KNNImputer`,
  * Scales all features with `StandardScaler`. 
* Computes complexity metrics using several **decomposition strategies**:

  * `OVO`, `OVA`, `OVO_closest`, `OVA_closest`, based on centroid distances.
* Optionally trains GPU-accelerated classifiers (RandomForest, kNN, Logistic Regression, GaussianNB, XGBoost) using stratified CV, measuring F1 and CD (mean probability assigned to the true class). 
* Stores:

  * Complexity results as nested arrays per metric,
  * CD scores,
  * Processed dataset IDs in the `RESULTS_PATH`.

These outputs are then used to populate `complexity_data/` for the regression scripts and `domain_analyzer.py`.

### `model_performance_Multiclass_analysis.py`

This script performs the regression study that underpins `GAMformerAnalysis`: 

* Loads `complexity_OVA*.npy` and `complexity_n_OVA.npy` together with `CD.npy`.
* Preprocesses them via `preprocess_complexity_data`, which:

  * Eliminates zero rows,
  * Computes final CD (1 − mean CD for each dataset),
  * Aggregates nested complexity arrays with a user-specified function (typically `np.mean`),
  * Replaces `NaN`s with zeros. 
* Defines multiple **feature subsets** (e.g., all metrics, F-only, L-only, instance-level, structural, imbalance) and evaluates:

  * **GAMformerRegressor** (GAMformer meta-model),
  * **ExplainableBoostingRegressor** (EBM).
* Uses `KFold` cross-validation (10 folds, fixed `RANDOM_STATE`) to compute:

  * Mean Absolute Error (MAE),
  * R² score,
  * Runtime per fold and feature set (wall-clock time around `fit` and `predict`). 
* Produces text + LaTeX-ready tables summarizing MAE (R²) and runtimes.

### `correlation_runtime_analysis.py`

This script compares different approximation levels of the complexity metrics (`C^{1/5}` … `C^{4/5}`) against the exact baseline (`complexity_n_OVA.npy`): 

* Loads complexity + CD data and computes Pearson correlations between each metric and CD.
* Computes relative percentage change in:

  * **Correlation performance** (Perf Δ%)
  * **Median computation time** (Time Δ%)
* Produces a publication-quality LaTeX table, using directional coloring (green = improvement, red = degradation) and booktabs formatting.

---

## Evaluation and Reproducibility

For reproducibility, all random states with respect to model testing are set identically to those utilized in the paper so that they yield the exact same results.

For benchmarking the difference between EBMs and GAMformer, you can run the `model_performance_Multiclass_analysis.py` with these settings:

```bash
sudo taskset -c 2 nice -n -15 "$(which conda)" run --live-stream --name gamformer python model_performance_Multiclass_analysis.py
```

**Command breakdown:**

* `sudo taskset -c 2`: Pins execution to CPU core 2 for consistent performance measurements
* `nice -n -15`: Sets high priority (low niceness value) for the process
* `"$(which conda)" run`: Uses conda to run in the specified environment
* `--live-stream`: Shows real-time output from the conda environment
* `--name gamformer`: Specifies the conda environment name

Combining this command with fixed random seeds in `model_performance_Multiclass_analysis.py` ensures that **both predictive performance and runtime numbers are exactly reproducible** on the same hardware setup. 

---

**Citation**

WIP

## License

This project is distributed under the terms specified in the `LICENSE` file in the repository root. Please consult that file before using this code in your own work.


