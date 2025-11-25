# -*- coding: utf-8 -*-


"""
Comparative Analysis of Approximation Variants (C^{k/5}) vs Standard Version

This script loads complexity measure results from approximation runs 
(C^{1/5} to C^{4/5}) and the exact version, computes Pearson correlation with 
Classification Difficulty (CD), and reports relative changes in:

• Correlation performance (Perf Δ %)
• Median computation time (Time Δ %)

The output is a publication-quality LaTeX table with directional coloring:
- Green: improvement (higher correlation or faster time)
- Red: degradation
using your exact original formatting and coloring logic.

Required LaTeX packages:
\\usepackage[table]{xcolor}
\\usepackage{booktabs}
\\usepackage{multirow}
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# -----------------------------
# --- PATHS & FILE DISCOVERY ---
# -----------------------------

RESULTS_DATA_PATH = os.path.join(os.getcwd(), 'complexity_data')
CD_FILE_PATH = os.path.join(RESULTS_DATA_PATH, 'CD.npy')

# Dynamically find all complexity files (same as your other scripts)
COMPLEXITY_FILES = [
    os.path.join(RESULTS_DATA_PATH, f)
    for f in os.listdir(RESULTS_DATA_PATH)
    if f.startswith('complexity_OVA') and f.endswith('.npy')
]
COMPLEXITY_FILES.append(os.path.join(RESULTS_DATA_PATH, "complexity_n_OVA.npy"))
COMPLEXITY_FILES.sort()

print("Found complexity files:")
for f in COMPLEXITY_FILES:
    print("  •", os.path.basename(f))

# Metric names (must match saved order)
METRICS = ['F2','F3','F4','L1','L2','L3','$R_{aug}$','N3','N4','N2','T1','$BI^{3}$']
RUNS = ['$\\mathbf{C^{1/5}}$', '$\\mathbf{C^{2/5}}$', '$\\mathbf{C^{3/5}}$', '$\\mathbf{C^{4/5}}$']

# -----------------------------
# --- CORRELATION FUNCTION ---
# -----------------------------

def get_correlation(data_complexity: np.ndarray, data_cd: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation for each metric with CD."""
    corr_final = np.zeros(data_complexity.shape[1])
    for k in range(data_complexity.shape[1]):
        if np.std(data_complexity[:, k]) == 0 or np.std(data_cd) == 0:
            corr_final[k] = 0.0
        else:
            r, _ = pearsonr(data_cd, data_complexity[:, k])
            corr_final[k] = r if np.isfinite(r) else 0.0
    return corr_final

# -----------------------------
# --- PREPROCESSING ---
# -----------------------------

def preprocess_file(filepath: str):
    data = np.load(filepath, allow_pickle=True)
    cd_data = np.load(CD_FILE_PATH, allow_pickle=True)

    values = data[0]   # complexity values
    times  = data[1]   # computation times

    # Remove failed runs
    valid = ~np.all(values == 0, axis=1) if values.ndim == 2 else ~np.all(np.all(v == 0 for v in row) for row in values)
    values, times, cd_data = values[valid], times[valid], cd_data[valid]

    # Final CD score
    cd_score = 1 - np.nanmean(cd_data, axis=1)

    # Aggregate nested repetitions
    X = np.zeros_like(values, dtype=float)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if not isinstance(val, int):
                X[i, j] = np.nanmean(val[0])
            else:
                X[i, j] = float(val) 
    X[np.isnan(X)] = 0.0

    return X, cd_score, times

# -----------------------------
# --- MAIN ANALYSIS ---
# -----------------------------

def main():
    print("\nStarting analysis...")

    # Load baseline (exact version)
    baseline_file = [f for f in COMPLEXITY_FILES if "complexity_n_OVA" in f][0]
    X_base, y_base, t_base = preprocess_file(baseline_file)
    corr_base = get_correlation(X_base, y_base)
    time_med_base = np.nanmedian(t_base.astype(np.float32), axis=0)

    # Load approximation variants
    approx_files = [f for f in COMPLEXITY_FILES if f != baseline_file]
    assert len(approx_files) == 4

    corr_list = []
    time_med_list = []

    for f in approx_files:
        X, y, t = preprocess_file(f)
        corr_list.append(get_correlation(X, y))
        time_med_list.append(np.nanmedian(t.astype(np.float32), axis=0))

    corr_approx = np.array(corr_list)
    time_approx = np.array(time_med_list)

    # Relative changes
    perf_diff = (corr_approx - corr_base) / np.abs(corr_base) * 100
    time_diff = np.zeros_like(time_approx)
    for i in range(4):
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = (time_approx[i] - time_med_base) / time_med_base * 100
            rel[~np.isfinite(rel)] = 0.0
            time_diff[i] = rel

    # Reorder 
    #perf_diff[[1, 2], :] = perf_diff[[2, 1], :]
    print(f"Shape perf_diff: {perf_diff.shape}")
    perf_diff[:,[6,9]] = perf_diff[:,[9,6]]

    # -----------------------------
    # TABLE FORMATTING ---
    # -----------------------------

    mat_time = pd.DataFrame(time_diff, index=RUNS, columns=METRICS)
    mat_perf = pd.DataFrame(perf_diff, index=RUNS, columns=METRICS)

    df = pd.concat([mat_time, mat_perf], axis=0).T
    df = df.round(3)

    def color_cells_by_value_2(x, thr1=5.0, thr2=10.0, fmt="{:.3f}"):
        try:
            val = float(x)
        except (ValueError, TypeError):
            return x
        s = fmt.format(val)
        if abs(val) >= 50:  # suppress huge values
            return s
        if val >= thr2:
            return f"\\cellcolor{{green!40}}{s}"
        elif val >= thr1:
            return f"\\cellcolor{{green!20}}{s}"
        elif val <= -thr2:
            return f"\\cellcolor{{red!40}}{s}"
        elif val <= -thr1:
            return f"\\cellcolor{{red!20}}{s}"
        else:
            return s

    df_styled = df.applymap(color_cells_by_value_2)

    # Exact original LaTeX export
    latex_table = df_styled.to_latex(
        buf=None,
        index=True,
        header=True,
        multicolumn=True,
        multicolumn_format="c",
        multirow=True,
        caption="Relative Time and Performance Differences Across Approximation Levels "
                "(vs. Standard Exact Version)",
        label="tab:approx_diffs",
        escape=False,
        column_format="l" + "r" * len(df.columns)
    ).replace("\\toprule", "\\toprule\n\\addlinespace") \
     .replace("\\midrule", "\\midrule\n\\addlinespace") \
     .replace("\\bottomrule", "\\bottomrule\n")

    print("\n" + "="*80)
    print("LATEX TABLE")
    print("="*80)
    print(latex_table)

    print("\nBaseline correlations:")
    print(pd.Series(np.abs(corr_base.round(4)), index=METRICS).to_string())

    print("\nDone.")

if __name__ == '__main__':
    main()