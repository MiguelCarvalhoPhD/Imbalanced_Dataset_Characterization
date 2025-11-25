# gamformer_analysis.py
# -*- coding: utf-8 -*-
"""
GAMformer Analysis object
Encapsulates loading meta-data, training a GAMformer meta-model,
computing complexity features for an input dataset, plotting analysis
and returning meta-model outputs.

Author: [Your Name]
Date: 2024-06-18 (refactor 2025-09-02)
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs, make_classification

# Local helper used in original script
from model_performance_Multiclass_analysis import preprocess_complexity_data

warnings.filterwarnings("ignore")

# Global feature names (kept for plotting & ordering)
FEATURE_NAMES = np.array(
    [
        "F2",
        "F3",
        "F4",
        "L1",
        "L2",
        "L3",
        "N2",
        "N3",
        "N4",
        "$R_{aug}$",
        "T1",
        "$BI^{3}$",
    ],
    dtype=object,
)


class GAMformerAnalysis:
    """
    Encapsulates the meta-model workflow:
      - load meta dataset (complexity features + CD targets)
      - fetch & fit (or load) a GAMformerRegressor meta-model
      - compute complexity metrics for a new dataset
      - plot analysis & return model outputs

    Usage:
        analysis = GAMformerAnalysis(
            complexity_file="path/to/complexity.npy",
            cd_file="path/to/CD.npy",
            model_str="my_model.cpkt",
        )
        analysis.plot_and_analyze((X, y, dataset_id), output_dir="out")
        outputs = analysis.get_meta_output(complexity_features)
    """

    def __init__(
        self,
        complexity_file: str,
        cd_file: str,
        model_str: str,
        aggregate_func: Callable = np.mean,
        device: Optional[str] = None,
    ):
        """
        Initialize: load meta dataset and train the meta-model.

        Args:
            complexity_file: path to .npy complexity meta-data
            cd_file: path to .npy class-difficulty (CD) file
            model_str: model checkpoint identifier (passed to fetch_model)
            aggregate_func: aggregation function for loading complexity arrays
            device: `'cuda'` or `'cpu'`, defaults to cuda if available
        """
        self.complexity_file = Path(complexity_file)
        self.cd_file = Path(cd_file)
        self.model_str = model_str
        self.aggregate_func = aggregate_func
        self.device = device
        self.meta_features: Optional[np.ndarray] = None
        self.meta_targets: Optional[np.ndarray] = None
        self.model: Optional[Any] = None

        # Try to import ticl inside init so import errors are captured when object is created
        try:
            from ticl.prediction import GAMformerRegressor  # type: ignore
            from ticl.utils import fetch_model  # type: ignore

            self._GAMformerRegressor = GAMformerRegressor
            self._fetch_model = fetch_model
        except Exception as exc:  # pragma: no cover - environment-specific
            raise ImportError(
                "The 'ticl' package is required but not available. "
                "Install it with `pip install ticl` or make sure it's on PYTHONPATH."
            ) from exc

        # Load meta-data and train model immediately (meta model training is always the same)
        self._load_meta_dataset()
        self._train_meta_model()

    # -------------------------
    # Internal helpers
    # -------------------------
    def _load_meta_dataset(self) -> None:
        """Loads and pre-processes meta dataset (calls your existing helper)."""
        if not self.complexity_file.exists() or not self.cd_file.exists():
            raise FileNotFoundError(
                f"Complexity file '{self.complexity_file}' or CD file '{self.cd_file}' not found."
            )

        # preprocess_complexity_data is assumed to return (X_meta, y_meta)
        X_meta, y_meta = preprocess_complexity_data(
            str(self.complexity_file), str(self.cd_file), self.aggregate_func
        )
        self.meta_features = X_meta
        self.meta_targets = y_meta
        print(
            f"[GAMformerAnalysis] Loaded meta dataset: features {X_meta.shape}, targets {y_meta.shape}"
        )

    def _train_meta_model(self) -> None:
        """Fetch and train a GAMformerRegressor on the meta dataset."""
        if self.meta_features is None or self.meta_targets is None:
            raise RuntimeError("Meta dataset not loaded.")

        # fetch checkpoint path using ticl.utils.fetch_model
        model_path = self._fetch_model(self.model_str)
        # determine device
        if self.device is None:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"

        # instantiate & fit
        print(f"[GAMformerAnalysis] Initializing GAMformer on device='{self.device}'...")
        self.model = self._GAMformerRegressor(device=self.device, path=model_path)
        print("[GAMformerAnalysis] Training / fitting meta-model...")
        self.model.fit(self.meta_features, self.meta_targets)
        print("[GAMformerAnalysis] Meta-model training complete.")

    # -------------------------
    # Public methods
    # -------------------------
    def compute_complexity_metrics(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute complexity metrics for an input dataset X, y.

        This method wraps the original compute_complexity_metrics function; it expects
        the same metric functions to be importable (as in your original script).
        """
        # Try to import required functions from your src package
        try:
            from src import (
                compute_F2_imbalanced_gpu,
                compute_F3_imbalanced_gpu,
                compute_F4_imbalanced_gpu,
                compute_L1_imbalanced_gpu,
                compute_L2_imbalanced_gpu,
                compute_L3_imbalanced_gpu,
                compute_N2_imbalanced_gpu,
                compute_N3_imbalanced_gpu,
                compute_N4_imbalanced_gpu,
                hypersphere_T1,
                compute_Raug_imbalanced_gpu,
                compute_bayes_imbalance_ratio,
            )
            from meta_dataset_extraction import one_vs_all_decomposition
        except Exception as exc:
            raise ImportError(
                "Could not import complexity metric functions (src.* or meta_dataset_extraction)."
            ) from exc

        complexity_functions = {
            "F2": compute_F2_imbalanced_gpu,
            "F3": compute_F3_imbalanced_gpu,
            "F4": compute_F4_imbalanced_gpu,
            "L1": compute_L1_imbalanced_gpu,
            "L2": compute_L2_imbalanced_gpu,
            "L3": compute_L3_imbalanced_gpu,
            "N2": compute_N2_imbalanced_gpu,
            "N3": compute_N3_imbalanced_gpu,
            "N4": compute_N4_imbalanced_gpu,
            "T1": hypersphere_T1,
            "Raug": compute_Raug_imbalanced_gpu,
            "BayesImbalance": compute_bayes_imbalance_ratio,
        }

        n_classes = len(np.unique(y))

        if n_classes > 2:
            complexity_metrics = np.zeros((len(complexity_functions), n_classes))
            for i, (name, func) in enumerate(complexity_functions.items()):
                values = one_vs_all_decomposition(func, X, y)
                complexity_metrics[i, :] = np.array(values[0])
        else:
            complexity_metrics = np.zeros((1, len(complexity_functions)))
            for i, (name, func) in enumerate(complexity_functions.items()):
                metric_values = func(X, y)
                if isinstance(metric_values, (list, tuple)):
                    complexity_metrics[0, i] = metric_values[-1]
                else:
                    complexity_metrics[0, i] = metric_values

        return complexity_metrics

    def get_meta_output(self, complexity_features: np.ndarray) -> Dict[str, Any]:
        """
        Run the trained meta-model on precomputed complexity features.

        Args:
            complexity_features: shape (1, n_features) expected.

        Returns:
            dict with keys: 'y_pred' (array), 'components' (array), 'sorted_features', 'sorted_components'
        """
        if self.model is None:
            raise RuntimeError("Meta-model has not been initialized or trained.")

        y_pred, components = self.model.predict_with_additive_components(complexity_features)
        # Flatten components into 1D array of contributions
        components = np.array([comp[0] for comp in components]).flatten()
        sorted_indices = np.argsort(np.abs(components))
        sorted_components = components[sorted_indices]
        sorted_features = FEATURE_NAMES[sorted_indices]

        return {
            "y_pred": np.array(y_pred),
            "components": components,
            "sorted_components": sorted_components,
            "sorted_features": sorted_features,
            "sorted_indices": sorted_indices,
        }

    def plot_and_analyze(self, full_dataset: Tuple[np.ndarray, np.ndarray, int], save_fig: bool, output_dir: str) -> None:
        """
        Compute complexity features for the dataset, predict with meta model,
        and generate a multipanel figure saved to disk.

        Args:
            full_dataset: (X, y, dataset_id)
            output_dir: directory path to save the figure
        """
        X_for_plot, y_for_plot, id_for_plot = full_dataset
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compute complexity metrics
        classwise_matrix = self.compute_complexity_metrics(X_for_plot, y_for_plot)
        is_multiclass = len(np.unique(y_for_plot)) > 2

        if is_multiclass:
            complexity_for_feat = np.nanmean(classwise_matrix, axis=1).reshape(1, -1)
        else:
            complexity_for_feat = classwise_matrix.reshape(1, -1)

        # Predict & get components
        outputs = self.get_meta_output(complexity_for_feat)
        y_pred = outputs["y_pred"]
        components = outputs["components"]
        sorted_indices = outputs["sorted_indices"]
        sorted_components = outputs["sorted_components"]
        sorted_features = outputs["sorted_features"]

        # Create figure
        n_cols = 3 if is_multiclass else 2
        width_ratios = [1.2, 0.7, 1] if is_multiclass else [1.2, 1]
        fig = plt.figure(figsize=(3.5 * n_cols, 3.5))
        gs = gridspec.GridSpec(1, n_cols, width_ratios=width_ratios, wspace=0.4)

        # Subplot 1: local explanations (horizontal bar)
        ax1 = fig.add_subplot(gs[0, 0])
        colors = ["#D10E0E" if x >= 0 else "#2ca02c" for x in sorted_components]
        ax1.barh(np.arange(len(sorted_components)), sorted_components, color=colors, edgecolor="k", linewidth=0.5, alpha=0.75, zorder=2)
        ax1.set(yticks=np.arange(len(sorted_features)), yticklabels=sorted_features)
        ax1.set_xlabel("Contribution to Prediction", fontsize=8)
        ax1.set_ylabel("Complexity Metric", fontsize=8)
        ax1.set_title(f"Local Explanation (Pred: {y_pred[0]:.2f})", fontweight="bold", fontsize=9)
        ax1.axvline(x=0, color="k", linestyle="--", alpha=0.5, zorder=0, linewidth=1)
        ax1.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)
        ax1.tick_params(axis="both", which="major", labelsize=7)

        # Subplot 2: class-wise complexity heatmap (if multiclass)
        if is_multiclass:
            ax2 = fig.add_subplot(gs[0, 1])
            sns.heatmap(classwise_matrix[np.flip(sorted_indices), :], cmap="viridis", ax=ax2, cbar=True, linecolor="k", linewidths=0.1, vmax=1, vmin=0, zorder=2, cbar_kws={"shrink": 0.7})
            cbar = ax2.collections[0].colorbar
            cbar.ax.tick_params(labelsize=6)
            ax2.set(yticks=np.arange(len(sorted_features)) + 0.5, yticklabels=np.flip(sorted_features, axis=0))
            ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=8)
            ax2.set(xticks=np.arange(len(np.unique(y_for_plot))) + 0.5, xticklabels=[f"{c}" for c in np.unique(y_for_plot)])
            ax2.set_xlabel("Class", fontsize=8)
            ax2.set_title("Class-wise Complexity", fontweight="bold", fontsize=9)
            ax2.tick_params(axis="x", which="major", labelsize=7)

        # Subplot 3: data visualization
        ax3 = fig.add_subplot(gs[0, -1])
        self._plot_data_2d(ax3, X_for_plot, y_for_plot, id_for_plot)

        fig.tight_layout()
        if save_fig:
            out_file = output_dir / f"analysis_dataset_{id_for_plot}.png"
            fig.savefig(out_file, dpi=400, bbox_inches="tight")
            print(f"[GAMformerAnalysis] Saved analysis to {out_file}")
        plt.show()
        plt.close(fig)
        
        return y_pred
        

    # -------------------------
    # static / small helpers
    # -------------------------
    @staticmethod
    def _plot_data_2d(ax: plt.Axes, X: np.ndarray, y: np.ndarray, dataset_id: int) -> None:
        """2D scatter with t-SNE if needed."""
        if X.shape[1] > 2:
            tsne = TSNE(n_components=2, perplexity=min(30, max(1, len(X) - 1)), random_state=42, init="random")
            X_2d = tsne.fit_transform(X)
            xlabel, ylabel = r"$t-SNE_1$", r"$t-SNE_2$"
        else:
            X_2d = X
            xlabel, ylabel = r"$Feature_1$", r"$Feature_2$"

        classes = np.unique(y)
        for c in classes:
            ax.scatter(X_2d[y == c, 0], X_2d[y == c, 1], label=f"Class {int(c)}", s=15, zorder=2, alpha=0.7)

        ax.set_title(f"2D Visualization (ID: {dataset_id})", fontweight="bold", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(edgecolor="k", loc="best", fontsize=6)
        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.5, zorder=0)


# -------------------------
# Example usage (run as script)
# -------------------------
if __name__ == "__main__":  # pragma: no cover - example
    # Default dataset and model paths (edit to suit your local paths)
    results_dir = Path.cwd() / "complexity_data"
    cd_file_multiclass = results_dir / "CD.npy"
    complexity_file_multiclass = results_dir / "complexity_n_OVA.npy"
    cd_file_binary = results_dir / "data_cd_binary.npy"
    complexity_file_binary = results_dir / "data_complexity_binary.npy"

    # Provide the model filename you use with ticl.fetch_model
    default_model_str = "baam_Daverage_l1e-05_maxnumclasses0_nsamples500_numfeatures10_yencoderlinear_05_08_2024_03_04_01_epoch_40.cpkt"

    #-----------------------------------
    #  BINARY DATASETS ANALYSIS 
    #-----------------------------------    
    
    # Create the object (this loads meta data and trains the meta-model)
    analysis = GAMformerAnalysis(
        complexity_file=str(complexity_file_binary),
        cd_file=str(cd_file_binary),
        model_str=default_model_str,
    )

    # Create a few synthetic datasets and run the plotting routine
    out_dir = "output_plots_synthetic"
    os.makedirs(out_dir, exist_ok=True)

    X1, y1 = make_blobs(n_samples=150, centers=2, n_features=2, random_state=42, cluster_std=1.5)
    X3, y3 = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=5, n_classes=2, random_state=1)

    for (X, y, idx) in [(X1, y1, 1), (X3, y3, 3)]:
        analysis.plot_and_analyze((X, y, idx), save_fig=False, output_dir=out_dir)

    #------------------------------------
    #   MULTICLASS DATASET ANALYSIS
    #-------------------------------------
    
    # Create the object 
    analysis = GAMformerAnalysis(
        complexity_file=str(complexity_file_multiclass),
        cd_file=str(cd_file_multiclass),
        model_str=default_model_str,
    )

    X2, y2 = make_blobs(n_samples=1500, centers=5, n_features=2, random_state=42, cluster_std=2.5)
    
    classification_difficulty = analysis.plot_and_analyze((X2,y2,2),save_fig=False,output_dir=out_dir)
