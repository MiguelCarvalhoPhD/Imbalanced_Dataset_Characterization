# -*- coding: utf-8 -*-

#%%
"""

This script performs a regression analysis to predict classification difficulty
(CD) based on pre-computed complexity measures. It evaluates different feature
subsets and regression models.

Methodology:
1.  **Data Loading & Preprocessing**:
    -   Loads complexity measure results and classification difficulty scores
        from specified `.npy` files.
    -   Applies a statistical aggregation function (e.g., mean, std) to the
        complexity results for each dataset.
    -   Calculates the final CD score (1 - mean CD).

2.  **Model Evaluation**:
    -   Defines several feature subsets based on groups of complexity measures.
    -   Uses k-fold cross-validation to train and evaluate regression models
        (e.g., GAMformer, Explainable Boosting Machine) on each feature subset.
    -   Records performance metrics (Mean Absolute Error, R2 Score) and model
        runtimes for each fold.

3.  **Results Formatting**:
    -   Averages the performance metrics across all folds for each file.
    -   Formats the results into a LaTeX-ready table, presenting MAE and R2
        scores together and highlighting the best-performing model for each
        feature set.
    -   Calculates and prints the average runtimes for comparison.

Dependencies:
-   numpy
-   scikit-learn
-   interpret
-   xgboost
-   ticl (for GAMformer)
-   matplotlib (optional, for plotting shape functions)
"""
import os
import time
import numpy as np
from typing import Dict, List, Callable, Tuple
import pandas as pd

# Machine Learning and Utility Libraries
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from interpret.glassbox import ExplainableBoostingRegressor
from ticl.prediction import GAMformerRegressor
from ticl.utils import fetch_model

# --- CONFIGURATION ---

# Paths and File Definitions
RESULTS_DATA_PATH = os.getcwd()+'/complexity_data/'
CD_FILE_PATH = os.path.join(RESULTS_DATA_PATH, 'CD.npy')

# Dynamically find complexity files to process
COMPLEXITY_FILES = [os.path.join(RESULTS_DATA_PATH, f) for f in os.listdir(RESULTS_DATA_PATH) if f.startswith('complexity_OVA')]
COMPLEXITY_FILES.append(os.path.join(RESULTS_DATA_PATH, "complexity_n_OVA.npy"))
COMPLEXITY_FILES.sort()

# Model and Cross-Validation Settings
N_SPLITS_CV = 10
RANDOM_STATE =  0
GAMFORMER_MODEL_STR = "baam_Daverage_l1e-05_maxnumclasses0_nsamples500_numfeatures10_yencoderlinear_05_08_2024_03_04_01_epoch_40.cpkt"

# Feature Set Definitions for Experiments
FEATURE_SETS = {
    'all': np.arange(12),
    'most_correlated': np.array([2, 5, 9, 10, 11]),
    'F-based': np.array([0, 1, 2]),
    'L-based': np.array([3, 4, 5]),
    'Instance-level': np.array([7, 8, 9]),
    'Structural-level': np.array([10, 6]),
    'Imbalance': np.array([11]),
    '1-F': np.array([2]),
    '1-L': np.array([5]),
    '1-I': np.array([9]),
    '1-S': np.array([10]),
}

# --- 1. DATA LOADING AND PREPROCESSING ---

def preprocess_complexity_data(
    complexity_filepath: str,
    cd_filepath: str,
    agg_func: Callable[[np.ndarray], float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads and preprocesses the complexity and classification difficulty data based
    on the user-provided structure.

    Args:
        complexity_filepath (str): Path to the complexity measures .npy file.
        cd_filepath (str): Path to the classification difficulty .npy file.
        agg_func (Callable): The function to aggregate complexity scores (e.g., np.mean).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - The preprocessed feature matrix (X).
            - The preprocessed target vector (y).
    """
    try:
        data_complexity = np.load(complexity_filepath, allow_pickle=True)
        data_cd = np.load(cd_filepath, allow_pickle=True)
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        return np.array([]), np.array([])

    # Preprocessing
    if data_complexity.ndim == 3:
        data_complexity = data_complexity[0, :, :]

        # Eliminate zero-rows
        valid_indices = ~np.all(data_complexity == 0, axis=1)
        data_complexity = data_complexity[valid_indices, :]
        data_cd = data_cd[valid_indices]

        # Compute final classification difficulty (CD)
        final_cd = 1 - np.nanmean(data_cd, axis=1)

        # Aggregate complexity measures using the provided function
        aggregated_complexity = np.zeros(data_complexity.shape)
        for row in range(data_complexity.shape[0]):
            for metric_idx in range(data_complexity.shape[1]):
                
                if not isinstance(data_complexity[row,metric_idx],int):
                    aggregated_complexity[row, metric_idx] = agg_func(data_complexity[row, metric_idx][0])
                else:
                    aggregated_complexity[row, metric_idx] = data_complexity[row, metric_idx]

        # Substitute NaNs with zeros
        aggregated_complexity[np.isnan(aggregated_complexity)] = 0
        
        return aggregated_complexity, final_cd
    
    else:
        # Binary data already preprocessed
        return data_complexity, data_cd

    


# --- 2. MODEL DEFINITION AND EXPERIMENT EXECUTION ---

def get_regression_models(gamformer_path: str) -> Dict[str, Callable]:
    """Returns a dictionary of regression models to be evaluated."""
    return {
        "GAMformer": GAMformerRegressor(device='cuda', path=gamformer_path),
        "EBM": ExplainableBoostingRegressor(max_bins=64,interactions=0, random_state=RANDOM_STATE)
    }

def run_regression_experiment(
    X: np.ndarray,
    y: np.ndarray,
    models: Dict[str, Callable],
    feature_sets: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the cross-validated regression experiment for all models and feature sets.

    Args:
        X (np.ndarray): The full feature matrix.
        y (np.ndarray): The target vector.
        models (Dict[str, Callable]): Dictionary of models to evaluate.
        feature_sets (Dict[str, np.ndarray]): Dictionary of feature subsets.

    Returns:
        Tuple containing mae_scores, r2_scores, and runtimes.
    """
    kf = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    num_models = len(models)
    num_feature_sets = len(feature_sets)

    mae_scores = np.zeros((num_models, num_feature_sets, kf.n_splits))
    r2_scores = np.zeros((num_models, num_feature_sets, kf.n_splits))
    runtimes = np.zeros((num_models, num_feature_sets, kf.n_splits))

    for fs_idx, (fs_name, features) in enumerate(feature_sets.items()):
        print(f"  Testing feature set: '{fs_name}'...")
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            X_train, X_test = X[train_idx][:, features], X[test_idx][:, features]
            y_train, y_test = y[train_idx], y[test_idx]

            for model_idx, (model_name, model_instance) in enumerate(models.items()):
                start_time = time.time()
                
                # Fit and predict
                model_instance.fit(X_train, y_train)
                y_pred = model_instance.predict(X_test)
                
                # Store results
                runtimes[model_idx, fs_idx, fold_idx] = time.time() - start_time
                mae_scores[model_idx, fs_idx, fold_idx] = mean_absolute_error(y_test, y_pred)
                r2_scores[model_idx, fs_idx, fold_idx] = r2_score(y_test, y_pred)
    
    return mae_scores, r2_scores, runtimes

# --- 3. RESULTS FORMATTING AND PRESENTATION ---

def format_results_for_latex(
    mean_mae: np.ndarray,
    mean_r2: np.ndarray,
    feature_set_names: List[str],
    model_names: List[str]
) -> pd.DataFrame:
    """
    Formats the regression results into a LaTeX-ready pandas DataFrame.

    Args:
        mean_mae (np.ndarray): Array of mean MAE scores.
        mean_r2 (np.ndarray): Array of mean R2 scores.
        feature_set_names (List[str]): Names of the feature sets used.
        model_names (List[str]): Names of the models evaluated.

    Returns:
        pd.DataFrame: A DataFrame with formatted strings for publication.
    """
    num_models, num_feature_sets = mean_mae.shape
    combined_results = np.empty((num_feature_sets, num_models), dtype=object)

    for fs_idx in range(num_feature_sets):
        # Find the best model for the current feature set based on R2 score
        best_model_idx = np.argmax(mean_r2[:, fs_idx])
        
        for model_idx in range(num_models):
            mae_val = mean_mae[model_idx, fs_idx]
            r2_val = mean_r2[model_idx, fs_idx]
            
            # Format as "MAE (R2)"
            value_str = f"{mae_val:.3f} ({r2_val:.3f})"
            
            # Bold the best performing model
            if model_idx == best_model_idx:
                value_str = f"\\textbf{{{value_str}}}"
            
            combined_results[fs_idx, model_idx] = value_str

    return pd.DataFrame(combined_results, index=feature_set_names, columns=model_names)

# --- MAIN EXECUTION ---

def main():
    """Main function to run the entire analysis pipeline."""
    print("Starting complexity analysis regression experiment...")

    gamformer_path = fetch_model(GAMFORMER_MODEL_STR)
    models = get_regression_models(gamformer_path)
    
    for file_path in COMPLEXITY_FILES:
        file_name = os.path.basename(file_path)
        print(f"\n{'='*60}")
        print(f"Processing file: {file_name}")
        print(f"{'='*60}")
        
        X, y = preprocess_complexity_data(file_path, CD_FILE_PATH, np.mean)
        if X.size == 0:
            print(f"  Skipping file due to loading/processing error.")
            continue
            
        mae, r2, runtimes = run_regression_experiment(X, y, models, FEATURE_SETS)
        
        # Average results across the folds for the current file
        mean_mae_for_file = np.mean(mae, axis=2)
        mean_r2_for_file = np.mean(r2, axis=2)
        mean_runtimes_for_file = np.mean(runtimes, axis=2)
        
        # Format and print the results table for the current file
        results_df = format_results_for_latex(
            mean_mae_for_file,
            mean_r2_for_file,
            list(FEATURE_SETS.keys()),
            list(models.keys())
        )
        
        print(f"\n--- Results for {file_name} (MAE (R2 Score)) ---")
        print(results_df.to_string())
        
        # Print runtime analysis for the current file
        print(f"\n--- Average Runtimes for {file_name} (seconds) ---")
        runtime_df = pd.DataFrame(mean_runtimes_for_file.T, index=list(FEATURE_SETS.keys()), columns=list(models.keys()))
        print(runtime_df.to_string())
        
        #print mean runtimes for EBM and GAMformer per file
        for model_idx, model_name in enumerate(models.keys()):
            avg_runtime = np.mean(mean_runtimes_for_file[model_idx, :])
            print(f"Average runtime for {model_name} on {file_name}: {avg_runtime:.4f} seconds")
            
        #print mean runtimes for EBM and GAMformer per file only for 'all' feature set
        for model_idx, model_name in enumerate(models.keys()):
            fs_idx = list(FEATURE_SETS.keys()).index('all')
            avg_runtime_all_fs = mean_runtimes_for_file[model_idx, fs_idx]
            print(f"Average runtime for {model_name} on {file_name} (all features): {avg_runtime_all_fs:.4f} seconds")

    print("\nAnalysis complete for all files.")


if __name__ == '__main__':
    main()

