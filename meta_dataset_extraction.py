# -*- coding: utf-8 -*-
"""
Publication-Ready Script for Multi-Class Complexity Analysis

This script provides a framework for analyzing the complexity of multi-class
classification problems. It implements various decomposition strategies
(One-vs-All, One-vs-One) to apply binary complexity measures to multi-class
datasets. The script is optimized for performance, leveraging GPU acceleration
via the RAPIDS cuML library where applicable.

Methodology:
1.  **Data Loading**: Loads datasets stored in NumPy's .npy format.
2.  **Preprocessing**: A robust scikit-learn pipeline handles missing values,
    encodes categorical features, and scales numerical features.
3.  **Complexity Measurement**:
    -   Decomposes multi-class problems into multiple binary problems using
        strategies like One-vs-One (OVO) and One-vs-All (OVA).
    -   Includes optimized OVO and OVA variants that consider only the
        most relevant or closest classes to reduce computational load.
    -   Applies a suite of user-defined binary complexity metrics to these
        sub-problems.
4.  **Model Evaluation**:
    -   Trains and evaluates a set of standard machine learning classifiers
        (e.g., RandomForest, k-NN, XGBoost) using cross-validation.
    -   Calculates performance metrics like F1-score and a custom
        Classification Difficulty (CD) metric.
5.  **Experiment Loop**: Iterates through multiple datasets, performs
    preprocessing, computes complexity measures, evaluates models, and
    saves the results for further analysis.

Dependencies:
-   numpy
-   scipy
-   scikit-learn
-   cuml (RAPIDS)
-   xgboost
-   matplotlib (optional, for visualization)
-   A local 'src.complexity_metrics' module containing the complexity functions.
"""

import os
import gc
import numpy as np
import scipy.sparse
from scipy.spatial import distance_matrix
from typing import Callable, List, Tuple

# Machine Learning and Preprocessing Libraries
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import f1_score

# GPU-accelerated libraries from RAPIDS cuML
from cuml.ensemble import RandomForestClassifier
from cuml.neighbors import KNeighborsClassifier
from cuml.linear_model import LogisticRegression
from cuml.naive_bayes import GaussianNB
from cuml.manifold import TSNE
from xgboost import XGBClassifier

# --- CONFIGURATION ---

# It's good practice to define paths and parameters at the top.
DATA_PATH = '/home/miguel_arvalho/PhD_tasks/1_1/multiclass_data/'
RESULTS_PATH = '/home/miguel_arvalho/complexity_measures/correlation_data/'
DATA_FILES = ['multi_data_new100.npy', 'multi_data_new200.npy', 'multi_data_new208.npy']
TARGET_FILES = ['multi_y_data_new100.npy', 'multi_y_data_new200.npy', 'multi_y_data_new208.npy']
N_SPLITS_CV = 5
MAX_INSTANCES = 500000  # Max dataset size to process
MAX_CLASSES = 30 # Max number of classes to process
GET_CD = True # Flag to enable/disable model evaluation for Classification Difficulty

# Define the decomposition strategy to be used for complexity calculation.
# Options: 'OVO', 'OVA', 'OVO_closest', 'OVA_closest'
DECOMPOSITION_STRATEGY = 'OVO_closest'

# --- 1. DATA PREPROCESSING ---

def build_preprocessor() -> Pipeline:
    """
    Constructs a scikit-learn pipeline for preprocessing tabular data.

    The pipeline performs the following steps:
    1.  Handles categorical features by imputing missing values with the most
        frequent value and then applying ordinal encoding.
    2.  Handles numerical features by imputing missing values using k-Nearest
        Neighbors.
    3.  Scales all features using StandardScaler.

    Returns:
        Pipeline: The configured preprocessing pipeline.
    """
    preprocessor = Pipeline([

    # Step 1: Handle categorical features
    ('cat_processing', ColumnTransformer([
        ('encode_cats', Pipeline([
            ('impute_cat', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder())
        ]), make_column_selector(dtype_include=['object', 'category']))
    ], remainder='passthrough')),
    
    # Step 3: Impute remaining missing values for numerical features
    ('impute_num', KNNImputer(n_neighbors=3)),

    # step 4: scaling
    ('scaling',StandardScaler())
    
    ])
    return preprocessor

def validate_dataset(X: np.ndarray, y: np.ndarray) -> bool:
    """
    Performs a series of checks to validate a dataset for the experiment.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        bool: True if the dataset is valid, False otherwise.
    """
    if X.shape[0] != len(y):
        print(f"Skipping dataset: Mismatch in number of samples. X has {X.shape[0]}, y has {len(y)}.")
        return False
    if X.shape[0] > MAX_INSTANCES:
        print(f"Skipping dataset: Exceeds maximum instance limit of {MAX_INSTANCES}.")
        return False
    if scipy.sparse.issparse(X) or scipy.sparse.issparse(y):
        print("Skipping dataset: Sparse matrices are not supported.")
        return False
    if len(y.shape) > 1:
        print("Skipping dataset: Multi-label classification is not supported.")
        return False
    if len(np.unique(y)) <= 1:
        print("Skipping dataset: Only one class present.")
        return False
    if len(np.unique(y)) > MAX_CLASSES:
        print(f"Skipping dataset: Exceeds maximum class limit of {MAX_CLASSES}.")
        return False
    # Check if any class has fewer than the number of CV splits
    min_samples_per_class = np.min(np.bincount(y.astype(int)))
    if GET_CD and min_samples_per_class < N_SPLITS_CV:
        print(f"Skipping dataset: At least one class has {min_samples_per_class} samples, but {N_SPLITS_CV} are required for CV.")
        return False
    return True

# --- 2. MULTI-CLASS DECOMPOSITION STRATEGIES ---

def one_vs_all_decomposition(metric: Callable, X: np.ndarray, y: np.ndarray) -> Tuple[List[float], int, float]:
    """
    Computes a complexity metric for all One-vs-All (OVA) binary subproblems.

    For each class, a binary problem is created where that class is the positive
    class (1) and all other classes are the negative class (0).

    Args:
        metric (Callable): A function that takes (X, y) for a binary problem
                           and returns a complexity score.
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target vector.

    Returns:
        Tuple[List[float], int, float]: A tuple containing:
            - A list of complexity scores, one for each class.
            - The number of classes.
            - The maximum imbalance ratio among the binary problems.
    """
    results = []
    lb = LabelBinarizer()
    binary_targets = lb.fit_transform(y)
    n_classes = binary_targets.shape[1]

    # Handle the binary case where LabelBinarizer returns a 1D array
    if n_classes == 1:
        binary_targets = np.hstack((1 - binary_targets, binary_targets))
        n_classes = 2

    for i in range(n_classes):
        complexity = metric(X, binary_targets[:, i])
        
        # Extract single float value from metric's output
        if isinstance(complexity, (list, tuple)):
            results.append(complexity[-1]) # Assuming last value is the main score
        else:
            results.append(float(complexity))

    class_counts = np.sum(binary_targets, axis=0)
    # Avoid division by zero if a class has no samples
    imbalance_ratio = np.max(class_counts) / np.min(class_counts[np.nonzero(class_counts)]) if np.min(class_counts) > 0 else np.inf

    return results, n_classes, imbalance_ratio

def one_vs_one_decomposition(metric: Callable, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes a complexity metric for all One-vs-One (OVO) binary subproblems.
    This is an efficient implementation that avoids redundant computations.

    Args:
        metric (Callable): The binary complexity metric function.
        X (np.ndarray): The feature matrix of the multi-class problem.
        y (np.ndarray): The target vector of the multi-class problem.

    Returns:
        np.ndarray: A symmetric matrix where `result[i, j]` holds the complexity
                    score for the binary problem between class `i` and class `j`.
    """
    classes = np.unique(y)
    n_classes = len(classes)
    ovo_matrix = np.full((n_classes, n_classes), np.nan)

    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            class1, class2 = classes[i], classes[j]

            # Create binary sub-problem
            indices = np.isin(y, [class1, class2])
            X_binary, y_binary = X[indices], y[indices]
            y_binary = np.where(y_binary == class1, 1, 0)

            # Compute and store metric
            complexity_score = metric(X_binary, y_binary)
            
            if isinstance(complexity_score, (list, tuple)):
                score = np.nanmean(complexity_score[:2])
            else:
                score = float(complexity_score)

            ovo_matrix[i, j] = ovo_matrix[j, i] = score

    return ovo_matrix

def one_vs_closest_rest_decomposition(metric: Callable, X: np.ndarray, y: np.ndarray, n_closest_func: Callable) -> List[float]:
    """
    Computes a complexity metric by decomposing the problem into multiple
    "One-vs-Closest-Rest" binary subproblems (referred to as OVA_closest).

    For each class, a binary problem is created where that class is positive (1)
    and the union of its 'k' nearest neighboring classes form the negative class (0).
    'k' is determined by `n_closest_func`.

    Args:
        metric (Callable): The binary complexity metric function.
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target vector.
        n_closest_func (Callable): A function that takes the number of classes
                                   and returns how many closest neighbors to consider.
                                   Example: `lambda n: int(np.sqrt(n))`.
    Returns:
        List[float]: A list of complexity scores, one for each decomposition.
    """
    results = []
    classes = np.unique(y)
    n_classes = len(classes)
    
    if n_classes <= 1:
        return []

    # Calculate class centroids
    class_centers = np.array([X[y == c].mean(axis=0) for c in classes])
    pairwise_distances = distance_matrix(class_centers, class_centers)
    np.fill_diagonal(pairwise_distances, np.inf)

    n_relevant_classes = n_closest_func(n_classes)
    # Ensure n_relevant_classes is at least 1 and less than n_classes
    n_relevant_classes = max(1, min(n_relevant_classes, n_classes - 1))

    for i, current_class in enumerate(classes):
        # Find the k closest classes (excluding the current one)
        closest_class_indices = np.argpartition(pairwise_distances[i, :], n_relevant_classes)[:n_relevant_classes]
        
        # Define the set of classes for the "Rest" group
        rest_classes = classes[closest_class_indices]
        
        # Create binary data subset
        relevant_indices = np.isin(y, np.append(rest_classes, current_class))
        X_binary, y_binary_original = X[relevant_indices], y[relevant_indices]
        
        # Relabel to {0, 1}
        y_binary = np.where(y_binary_original == current_class, 1, 0)

        complexity = metric(X_binary, y_binary)
        
        if isinstance(complexity, (list, tuple)):
            results.append(complexity[-2]) # Assuming second to last value is main score
        else:
            results.append(float(complexity))
            
    return results

def one_vs_one_closest_decomposition(metric: Callable, X: np.ndarray, y: np.ndarray,
                                     n_closest_func: Callable) -> np.ndarray:
    """
    Computes a complexity metric for OVO subproblems, but only for the closest
    pairs of classes, determined by the distance between class centroids.

    Args:
        metric (Callable): The binary complexity metric function.
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target vector.
        n_closest_func (Callable): A function that takes the number of classes
                                   and returns how many closest neighbors to consider.
                                   Example: `lambda n: int(np.sqrt(n))`.

    Returns:
        np.ndarray: A matrix containing complexity scores for the analyzed pairs.
                    Unanalyzed pairs are marked with NaN.
    """
    classes = np.unique(y)
    n_classes = len(classes)
    
    if n_classes <= 1:
        return np.array([[]])
        
    # Calculate class centroids
    class_centers = np.array([X[y == c].mean(axis=0) for c in classes])
    pairwise_distances = distance_matrix(class_centers, class_centers)
    np.fill_diagonal(pairwise_distances, np.inf) # Ignore self-distance

    ovo_matrix = np.full((n_classes, n_classes), np.nan)
    n_neighbors = n_closest_func(n_classes)
    if n_neighbors >= n_classes:
        n_neighbors = n_classes - 1

    for i in range(n_classes):
        # Find the indices of the n closest classes
        closest_indices = np.argpartition(pairwise_distances[i, :], n_neighbors)[:n_neighbors]

        for j in closest_indices:
            if i >= j: continue # Avoid redundant calculations

            class1, class2 = classes[i], classes[j]
            indices = np.isin(y, [class1, class2])
            X_binary, y_binary = X[indices], y[indices]
            y_binary = np.where(y_binary == class1, 1, 0)

            complexity_score = metric(X_binary, y_binary)
            
            if isinstance(complexity_score, (list, tuple)):
                score = np.nanmean(complexity_score[:2])
            else:
                score = float(complexity_score)

            ovo_matrix[i, j] = ovo_matrix[j, i] = score
            
    return ovo_matrix


# --- 3. CLASSIFICATION AND PERFORMANCE EVALUATION ---

def get_classifiers() -> dict:
    """Returns a dictionary of cuML-based classifiers for evaluation."""
    return {
        'RandomForest': RandomForestClassifier(),
        'kNN': KNeighborsClassifier(),
        'LogisticRegression': LogisticRegression(max_iter=10*5),
        'GaussianNB': GaussianNB(),
        'XGBoost': XGBClassifier(device='cuda', random_state=42)
    }

def evaluate_models_cross_val(X: np.ndarray, y: np.ndarray, n_splits: int) -> dict:
    """
    Trains and evaluates multiple classifiers using stratified cross-validation.

    Args:
        X (np.ndarray): The preprocessed feature matrix.
        y (np.ndarray): The encoded target vector.
        n_splits (int): The number of folds for cross-validation.

    Returns:
        dict: A dictionary mapping model names to their average macro F1-score
              and average Classification Difficulty (CD) across folds.
    """
    classifiers = get_classifiers()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = {name: {'f1_macro': [], 'cd': []} for name in classifiers}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        for name, model in classifiers.items():
            try:
                fresh_model = get_classifiers()[name] # Re-initialize for clean state
                fresh_model.fit(X_train.astype(np.float32), y_train)
                
                y_pred_proba = fresh_model.predict_proba(X_test.astype(np.float32))
                y_pred_labels = fresh_model.predict(X_test.astype(np.float32))

                f1 = f1_score(y_test, y_pred_labels, average='macro', zero_division=0)
                results[name]['f1_macro'].append(f1)

                # CD is the average probability assigned to the true class.
                true_class_probs = y_pred_proba[np.arange(len(y_test)), y_test]
                results[name]['cd'].append(np.mean(true_class_probs))

            except Exception as e:
                print(f"Error evaluating {name} on fold {fold}: {e}")
                results[name]['f1_macro'].append(np.nan)
                results[name]['cd'].append(np.nan)

    # Average the results across folds
    avg_results = {}
    for name, scores in results.items():
        avg_results[name] = {
            'f1_macro_mean': np.nanmean(scores['f1_macro']),
            'cd_mean': np.nanmean(scores['cd'])
        }
        
    return avg_results

# --- 4. VISUALIZATION (Optional) ---
def visualize_with_tsne(X: np.ndarray, y: np.ndarray, title: str = 't-SNE Visualization'):
    """
    Generates and displays a 2D t-SNE plot of the data.
    This requires matplotlib to be installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not found. Skipping visualization.")
        return

    print("Generating t-SNE plot...")
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(X.astype(np.float32))

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(handles=scatter.legend_elements()[0], labels=np.unique(y).tolist())
    plt.grid(True)
    plt.show()


# --- 5. MAIN EXPERIMENT EXECUTION ---

def run_experiment():
    """
    Main function to run the entire experimental pipeline.
    """
    # --- Setup ---
    try:
        from src import (
            compute_F2_imbalanced_gpu, compute_F3_imbalanced_gpu,
            compute_F4_imbalanced_gpu, compute_L1_imbalanced_gpu,
            compute_L2_imbalanced_gpu, compute_L3_imbalanced_gpu,
            compute_N2_imbalanced_gpu, compute_N3_imbalanced_gpu,
            compute_N4_imbalanced_gpu, hypersphere_T1,
            compute_Raug_imbalanced_gpu, compute_bayes_imbalance_ratio
        )
        print("Successfully imported complexity metrics from 'src.complexity_metrics'.")

    except ImportError:
        print("Warning: 'src.complexity_metrics' not found. Using placeholder metrics.")
        def placeholder_metric(X, y): return np.random.rand()
        compute_F2_imbalanced_gpu = compute_F3_imbalanced_gpu = \
        compute_F4_imbalanced_gpu = compute_L1_imbalanced_gpu = \
        compute_L2_imbalanced_gpu = compute_L3_imbalanced_gpu = \
        compute_N2_imbalanced_gpu = compute_N3_imbalanced_gpu = \
        compute_N4_imbalanced_gpu = hypersphere_T1 = \
        compute_Raug_imbalanced_gpu = compute_bayes_imbalance_ratio = placeholder_metric

    complexity_functions = {
        'F2': compute_F2_imbalanced_gpu, 'F3': compute_F3_imbalanced_gpu,
        'F4': compute_F4_imbalanced_gpu, 'L1': compute_L1_imbalanced_gpu,
        'L2': compute_L2_imbalanced_gpu, 'L3': compute_L3_imbalanced_gpu,
        'N2': compute_N2_imbalanced_gpu, 'N3': compute_N3_imbalanced_gpu,
        'N4': compute_N4_imbalanced_gpu, 'T1': hypersphere_T1,
        'Raug': compute_Raug_imbalanced_gpu,
        'BayesImbalance': compute_bayes_imbalance_ratio
    }
    
    preprocessor = build_preprocessor()
    label_encoder = LabelEncoder()
    
    all_complexity_results = []
    all_cd_results = []
    processed_dataset_ids = []

    # --- Data Loading and Iteration ---
    for data_file, target_file in zip(DATA_FILES, TARGET_FILES):
        try:
            features_data = np.load(os.path.join(DATA_PATH, data_file), allow_pickle=True)
            targets_data = np.load(os.path.join(DATA_PATH, target_file), allow_pickle=True)
        except FileNotFoundError:
            print(f"Warning: Could not find {data_file} or {target_file}. Skipping.")
            continue

        dataset_ids = targets_data[1, :]
        targets_list = targets_data[0, :]

        for i, (X_raw, y_raw, dataset_id) in enumerate(zip(features_data, targets_list, dataset_ids)):
            print("-" * 60)
            print(f"Processing Dataset ID: {dataset_id} ({i+1}/{len(dataset_ids)})")

            try:
                # --- Preprocessing and Validation ---
                y_enc = label_encoder.fit_transform(y_raw)
                if not validate_dataset(X_raw, y_enc):
                    continue
                X_proc = preprocessor.fit_transform(X_raw, y_enc)

                # --- Complexity Calculation ---
                current_complexity_row = []
                for metric_name, metric_func in complexity_functions.items():
                    
                    if DECOMPOSITION_STRATEGY == 'OVO_closest':
                        result = one_vs_one_closest_decomposition(metric_func, X_proc, y_enc, lambda n: int(np.sqrt(n)))
                    elif DECOMPOSITION_STRATEGY == 'OVO':
                        result = one_vs_one_decomposition(metric_func, X_proc, y_enc)
                    elif DECOMPOSITION_STRATEGY == 'OVA':
                        result, _, _ = one_vs_all_decomposition(metric_func, X_proc, y_enc)
                    elif DECOMPOSITION_STRATEGY == 'OVA_closest':
                        result = one_vs_closest_rest_decomposition(metric_func, X_proc, y_enc, lambda n: int(np.sqrt(n)))
                    
                    else:
                        raise ValueError(f"Unknown decomposition strategy: {DECOMPOSITION_STRATEGY}")
                    
                    print(f"Computed {metric_name} for dataset {dataset_id}: {np.nanmean(result):.4f}")
                    
                    current_complexity_row.append(result)
                all_complexity_results.append(current_complexity_row)
                
                # --- Model Performance Evaluation (optional) ---
                if GET_CD:
                    model_performance = evaluate_models_cross_val(X_proc, y_enc, n_splits=N_SPLITS_CV)
                    current_cd_row = [model_performance[name]['cd_mean'] for name in get_classifiers().keys()]
                    all_cd_results.append(current_cd_row)
                
                processed_dataset_ids.append(dataset_id)

            except Exception as e:
                print(f"An error occurred while processing dataset {dataset_id}: {e}")
                continue
            
            gc.collect()

    # --- Save Results ---
    if processed_dataset_ids:
        os.makedirs(RESULTS_PATH, exist_ok=True)

        complexity_output_path = os.path.join(RESULTS_PATH, f'complexity_results_{DECOMPOSITION_STRATEGY}.npy')
        np.save(complexity_output_path, np.array(all_complexity_results, dtype=object), allow_pickle=True)
        print(f"\nSaved complexity results to {complexity_output_path}")

        ids_output_path = os.path.join(RESULTS_PATH, 'processed_dataset_ids.npy')
        np.save(ids_output_path, np.array(processed_dataset_ids))
        print(f"Saved dataset IDs to {ids_output_path}")

        if GET_CD and all_cd_results:
            cd_output_path = os.path.join(RESULTS_PATH, 'classification_difficulty_scores.npy')
            np.save(cd_output_path, np.array(all_cd_results))
            print(f"Saved classification difficulty scores to {cd_output_path}")
            
    else:
        print("\nExperiment finished. No valid datasets were processed.")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    run_experiment()
