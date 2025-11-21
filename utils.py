from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from itertools import combinations
from sklearn.metrics import ndcg_score, jaccard_score, roc_auc_score
from typing import Callable, Dict, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from collections import defaultdict


def compute_pairwise_metric(
    results_dict: Dict[str, Dict[str, list]],
    key: str,
    metric_fn: Callable[[Any, Any], float],
) -> Tuple[np.ndarray, list]:
    """
    Generic function to compute pairwise similarity/distance matrices
    across models for a given metric and key (e.g., 'shapvalues', 'scores', 'predictions').
    """
    model_names = sorted(results_dict.keys() - {"ground_truth"})
    n_models = len(model_names)
    n_fold = len(results_dict[model_names[0]])

    # Préchargement en mémoire pour éviter les lookups répétés
    data = {m: [results_dict[m][f][key] for f in range(n_fold)] for m in model_names}

    matrix = np.eye(n_models, dtype=float)

    for i, j in combinations(range(n_models), 2):
        # Calcul vectorisé sur les folds si metric_fn n’est pas vectorisable
        scores = [
            metric_fn(data[model_names[i]][f], data[model_names[j]][f])
            for f in range(n_fold)
        ]
        matrix[i, j] = matrix[j, i] = np.nanmean(scores)

    return matrix, model_names


# --- Metric-specific wrappers --------------------------------------------------


def pearson_corr_vectorized(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute per-sample Pearson correlation between x and y.
    x, y shape: (n_samples, n_features)
    Returns: array of correlations (n_samples,)
    """
    x_mean = x.mean(axis=1, keepdims=True)
    y_mean = y.mean(axis=1, keepdims=True)
    num = np.sum((x - x_mean) * (y - y_mean), axis=1)
    den = np.sqrt(np.sum((x - x_mean) ** 2, axis=1) * np.sum((y - y_mean) ** 2, axis=1))
    return num / np.maximum(den, 1e-12)


def compute_shap_similarity_pearson(
    results_dict: Dict[str, Dict[str, list]],
) -> Tuple[np.ndarray, list]:
    """Compute pairwise Pearson correlation similarity between SHAP values."""

    return compute_pairwise_metric(results_dict, "shapvalues", pearson_corr_vectorized)


def compute_ndcg_similarity(
    results_dict: Dict[str, Dict[str, list]],
) -> Tuple[np.ndarray, list]:
    """Compute pairwise NDCG similarity between SHAP value magnitudes."""

    def ndcg_metric(model_i_fold, model_j_fold) -> float:
        if len(model_i_fold.shape) == 1:
            model_i_fold = np.array([model_i_fold])
        if len(model_j_fold.shape) == 1:
            model_j_fold = np.array([model_j_fold])

        score_ij = ndcg_score(np.abs(model_i_fold), np.abs(model_j_fold))
        score_ji = ndcg_score(np.abs(model_j_fold), np.abs(model_i_fold))
        return np.mean([score_ij, score_ji])  # type: ignore

    return compute_pairwise_metric(results_dict, "shapvalues", ndcg_metric)


def compute_score_correlations(
    results_dict: Dict[str, Dict[str, list]],
) -> Tuple[np.ndarray, list]:
    """Compute Pearson correlation between models' anomaly scores."""

    def score_corr(model_i_fold, model_j_fold) -> float:
        return np.corrcoef(model_i_fold, model_j_fold)[0, 1]

    return compute_pairwise_metric(results_dict, "scores", score_corr)


def compute_pred_jaccard(
    results_dict: Dict[str, Dict[str, list]],
) -> Tuple[np.ndarray, list]:
    """Compute pairwise Jaccard index between binary predictions."""

    def jaccard_metric(model_i_fold, model_j_fold) -> float:
        return jaccard_score(
            model_i_fold,
            model_j_fold,
            zero_division=0,
            average="binary",
            pos_label=1,
        )  # type: ignore

    return compute_pairwise_metric(results_dict, "predictions", jaccard_metric)


def compute_auc_roc(results_dict):
    model_names = list(set((results_dict.keys())) - set(["ground_truth"]))
    n_models = len(model_names)
    n_fold = len(results_dict[model_names[0]])

    all_mccs = []

    for model in range(n_models):
        mccs = []
        for fold_idx in range(n_fold):
            y_true = results_dict["ground_truth"][fold_idx]
            y_test = results_dict[model_names[model]][fold_idx]["scores"]

            mccs.append(roc_auc_score(y_true, y_test))

        all_mccs.append(np.nanmean(np.array(mccs)))

    return all_mccs, model_names


def plot_heatmaps(
    matrices: Dict[str, pd.DataFrame],
    figsize=(20, 7),
    colormaps=["OrRd", "Blues", "YlGn", "PuBu"],
    n_colors=6,
):
    """Plot all matrices side by side as heatmaps."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for ax, (name, df), cmap_name in zip(axes.ravel(), matrices.items(), colormaps):

        # Discretisation of the colormap
        base_cmap = plt.cm.get_cmap(cmap_name)
        colors = base_cmap(np.linspace(0, 1, n_colors))
        discrete_cmap = ListedColormap(colors)

        vmin, vmax = np.min(df.values), np.max(df.values)
        bounds = np.linspace(vmin, vmax, n_colors + 1)
        norm = BoundaryNorm(bounds, discrete_cmap.N)

        sns.heatmap(
            df,
            cmap=base_cmap,
            annot=True,
            fmt=".0f",
            linewidths=0.25,
            linecolor="black",
            cbar=False,
            annot_kws={"size": 8},
            ax=ax,
            # norm=norm,
        )
        ax.set_title(name, fontsize=13)

    plt.tight_layout()
    # plt.show()
    return fig


def load_nested_results(base_path) -> Dict[str, Any]:
    """
    Charge récursivement les résultats sérialisés (pickle) à partir d'une structure de dossiers
    et les organise dans un dictionnaire imbriqué standard.

    La structure attendue est : base_path/dataset/model/fold.pkl

    Args:
        base_path: Chemin d'accès racine contenant les dossiers de datasets.

    Returns:
        Un dictionnaire standard (dict) imbriqué contenant les résultats chargés.
    """
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # Parcourir les dossiers de datasets
    for dataset in sorted(os.listdir(base_path)):
        dataset_path = os.path.join(base_path, dataset)

        # Vérifier si c'est un dossier et ignorer les fichiers cachés
        if not os.path.isdir(dataset_path) or dataset.startswith("."):
            continue

        # Parcourir les dossiers de modèles
        for model in sorted(os.listdir(dataset_path)):
            model_path = os.path.join(dataset_path, model)

            if not os.path.isdir(model_path) or model.startswith("."):
                continue

            # Parcourir les fichiers de fold (résultats)
            for fold_file in sorted(os.listdir(model_path)):
                if fold_file.startswith(".") or not fold_file.endswith(".pkl"):
                    continue

                result_path = os.path.join(model_path, fold_file)

                # Le 'fold_file' est supposé être de la forme 'N.pkl'
                try:
                    fold_number = int(fold_file[:-4])
                except ValueError:
                    # Ignorer les fichiers qui ne correspondent pas au format attendu (N.pkl)
                    continue

                with open(result_path, "rb") as f:
                    # Charger les résultats et les stocker dans le defaultdict imbriqué
                    all_results[dataset][model][fold_number] = pickle.load(f)

    # Convertir le defaultdict imbriqué en dict standard
    def to_regular_dict(d):
        if isinstance(d, dict):
            return {k: to_regular_dict(v) for k, v in d.items()}
        return d

    return to_regular_dict(all_results)
