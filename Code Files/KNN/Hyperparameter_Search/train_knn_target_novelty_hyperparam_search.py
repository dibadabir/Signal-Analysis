# ============================================================
# train_knn_target_novelty_hyperparam_search.py
# ============================================================
# KNN Hyperparameter Search on Target + Novelty Concatenated Preprocessing
# Purpose: Find optimal K, weights, and feature count with subject-aware CV
# 5-Fold StratifiedGroupKFold Cross-Validation
# CRITICAL: Feature selection and scaling happen INSIDE CV folds
#           to prevent data leakage and ensure unbiased estimates
# ============================================================

import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# ============================================================
# Configuration
# ============================================================
FEATURES_PATH = Path("outputs") / "FEATURES_KNN_TARGET_NOVELTY_CONCAT.npz"
OUT_DIR = Path("outputs") / "knn_hyperparam_search_target_novelty"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Hyperparameter Search Space
# ============================================================
HYPERPARAMETER_GRID = {
    "n_features": [40, 60, 80],
    "k": [9, 11, 13, 15],
    "weights": ["uniform", "distance"],
    "metric": ["manhattan"],  # Fixed
}

# Cross-validation
N_SPLITS = 5

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger("knn_hyperparam_search")


# ============================================================
# Helper Functions
# ============================================================

def calculate_metrics(y_true, y_pred, y_proba=None):
    """Calculate comprehensive performance metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),  # TP / (TP + FN)
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    # Confusion matrix based metrics (use labels=[0,1] for robustness)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics["specificity"] = specificity
    metrics["false_negative_rate"] = fnr

    # Safely compute ROC AUC
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan

    return metrics


def select_features_inside_fold(X_train, y_train, X_val, n_features=60):
    """
    Select top features using mutual information on TRAINING data only.
    
    CRITICAL FOR AVOIDING LEAKAGE:
    - Feature importance is computed from training data only
    - The same selected features are applied to validation data
    - No information about validation data is used in selection
    
    Returns:
    - X_train_selected: (n_train, n_features)
    - X_val_selected: (n_val, n_features)
    - selected_indices: indices of selected features
    - mi_scores: mutual information scores for selected features
    """
    # Ensure n_features doesn't exceed available features
    n_features = min(n_features, X_train.shape[1])
    
    # Compute mutual information on TRAINING DATA ONLY
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    
    # Select top N features based on training MI scores
    top_indices = np.argsort(mi_scores)[-n_features:][::-1]
    selected_mi_scores = mi_scores[top_indices]
    
    # Apply same feature selection to both train and validation
    X_train_selected = X_train[:, top_indices]
    X_val_selected = X_val[:, top_indices]
    
    return X_train_selected, X_val_selected, top_indices, selected_mi_scores


def scale_inside_fold(X_train, X_val):
    """
    Scale features using StandardScaler FIT ON TRAINING DATA ONLY.
    
    CRITICAL FOR AVOIDING LEAKAGE:
    - Scaler statistics (mean, std) are computed from training data only
    - Validation data is transformed using training statistics
    - No information about validation data affects the scaler
    
    Returns:
    - X_train_scaled: standardized training data
    - X_val_scaled: standardized validation data
    - scaler: the fitted StandardScaler object
    """
    scaler = StandardScaler()
    
    # Fit scaler on TRAINING DATA ONLY
    scaler.fit(X_train)
    
    # Apply same transformation to both train and validation
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    return X_train_scaled, X_val_scaled, scaler


def evaluate_hyperparameter_combination(
    X, y, groups, n_features, k, weights, metric
) -> Dict:
    """
    Evaluate a single hyperparameter combination using 5-fold StratifiedGroupKFold.
    
    Returns a dictionary with mean/std metrics across folds.
    """
    fold_accuracies = []
    fold_sensitivities = []
    fold_specificities = []
    fold_precisions = []
    fold_f1s = []
    fold_aucs = []

    # Cross-validation setup
    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # Cross-validation loop
    for train_idx, val_idx in skf.split(X, y, groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Feature selection inside fold (using training data only)
        X_train_sel, X_val_sel, _, _ = select_features_inside_fold(
            X_train, y_train, X_val, n_features=n_features
        )

        # Scaling inside fold (fit on training data only)
        X_train_scaled, X_val_scaled, _ = scale_inside_fold(X_train_sel, X_val_sel)

        # KNN training
        model = KNeighborsClassifier(
            n_neighbors=k,
            metric=metric,
            weights=weights,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_val_pred = model.predict(X_val_scaled)
        y_val_proba = model.predict_proba(X_val_scaled)[:, 1]

        # Metrics
        val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)

        fold_accuracies.append(val_metrics["accuracy"])
        fold_sensitivities.append(val_metrics["sensitivity"])
        fold_specificities.append(val_metrics["specificity"])
        fold_precisions.append(val_metrics["precision"])
        fold_f1s.append(val_metrics["f1_score"])
        fold_aucs.append(val_metrics["roc_auc"])

    # Aggregate results
    result = {
        "n_features": n_features,
        "k": k,
        "weights": weights,
        "metric": metric,
        "mean_accuracy": np.mean(fold_accuracies),
        "std_accuracy": np.std(fold_accuracies),
        "mean_sensitivity": np.mean(fold_sensitivities),
        "std_sensitivity": np.std(fold_sensitivities),
        "mean_specificity": np.mean(fold_specificities),
        "mean_precision": np.mean(fold_precisions),
        "mean_f1": np.mean(fold_f1s),
        "mean_auc": np.nanmean(fold_aucs),
        "std_auc": np.nanstd(fold_aucs),
    }

    return result


def plot_hyperparameter_heatmaps(results_df: pd.DataFrame):
    """Create heatmaps showing performance across hyperparameters."""
    try:
        # Group by features and weights for K sensitivity analysis
        for n_feat in sorted(results_df["n_features"].unique()):
            for w in sorted(results_df["weights"].unique()):
                subset = results_df[
                    (results_df["n_features"] == n_feat) & (results_df["weights"] == w)
                ]
                if len(subset) == 0:
                    continue

                # Create pivot table for heatmap
                pivot_data = subset.pivot_table(
                    values="mean_sensitivity",
                    index="k",
                    aggfunc="first"
                )

                fig, ax = plt.subplots(figsize=(8, 4))
                sns.heatmap(
                    pivot_data, annot=True, fmt=".3f", cmap="RdYlGn",
                    ax=ax, cbar_kws={"label": "Mean Sensitivity"}
                )
                ax.set_title(
                    f"Mean Sensitivity by K (features={n_feat}, weights={w})",
                    fontsize=12, fontweight="bold"
                )

                plot_path = PLOTS_DIR / f"sensitivity_heatmap_f{n_feat}_w{w}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                logger.info(f"  ✓ Saved heatmap: {plot_path}")
                plt.close()

    except Exception as e:
        logger.warning(f"  ⚠️  Could not generate heatmaps: {e}")


def rank_configurations(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank configurations by priority:
    1. Highest mean sensitivity (recall for PD)
    2. Good ROC-AUC
    3. Good accuracy
    4. Lower standard deviation across folds
    """
    # Create ranking score
    # Higher sensitivity = better (weight: 0.5)
    # Higher AUC = better (weight: 0.3)
    # Higher accuracy = better (weight: 0.15)
    # Lower std_sensitivity = better (weight: 0.05) - favor stability
    
    max_sens = results_df["mean_sensitivity"].max()
    max_auc = results_df["mean_auc"].max()
    max_acc = results_df["mean_accuracy"].max()
    max_std = results_df["std_sensitivity"].max()

    results_df["rank_score"] = (
        0.50 * (results_df["mean_sensitivity"] / max_sens) +
        0.30 * (results_df["mean_auc"] / max_auc) +
        0.15 * (results_df["mean_accuracy"] / max_acc) +
        0.05 * ((max_std - results_df["std_sensitivity"]) / max_std)
    )

    # Sort by rank score descending
    ranked = results_df.sort_values("rank_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)

    return ranked


# ============================================================
# Main Search Function
# ============================================================

def main():
    logger.info("=" * 80)
    logger.info("KNN HYPERPARAMETER SEARCH - TARGET + NOVELTY CONCATENATED PREPROCESSING")
    logger.info("=" * 80)

    # Load features
    logger.info(f"\nLoading features from: {FEATURES_PATH}")
    data = np.load(FEATURES_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    
    # Safely extract feature mode
    if "feature_mode" in data.files:
        feature_mode = str(data["feature_mode"])
    else:
        feature_mode = "unknown"
        logger.warning("  ⚠️  Feature mode not found in npz file")

    # Dataset summary
    logger.info(f"\n{'=' * 80}")
    logger.info("DATASET SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(f"Feature mode: {feature_mode}")
    logger.info(f"Total epochs: {X.shape[0]}")
    logger.info(f"Input features: {X.shape[1]}")
    logger.info(f"Total subjects: {len(np.unique(groups))}")
    
    n_pd = np.sum(y == 1)
    n_ctl = np.sum(y == 0)
    logger.info(f"Classes: {n_pd} PD, {n_ctl} CTL ({n_pd/len(y):.1%} PD)")

    # Generate all hyperparameter combinations
    logger.info(f"\n{'=' * 80}")
    logger.info("HYPERPARAMETER SEARCH CONFIGURATION")
    logger.info(f"{'=' * 80}")
    logger.info(f"n_features: {HYPERPARAMETER_GRID['n_features']}")
    logger.info(f"k: {HYPERPARAMETER_GRID['k']}")
    logger.info(f"weights: {HYPERPARAMETER_GRID['weights']}")
    logger.info(f"metric: {HYPERPARAMETER_GRID['metric']} (fixed)")
    logger.info(f"cv_strategy: {N_SPLITS}-Fold StratifiedGroupKFold")

    # Generate combinations
    combinations = []
    for n_feat in HYPERPARAMETER_GRID["n_features"]:
        for k in HYPERPARAMETER_GRID["k"]:
            for w in HYPERPARAMETER_GRID["weights"]:
                for m in HYPERPARAMETER_GRID["metric"]:
                    combinations.append((n_feat, k, w, m))

    n_combinations = len(combinations)
    logger.info(f"Total combinations to evaluate: {n_combinations}")
    logger.info(f"\nLeakage prevention:")
    logger.info(f"  ✓ Feature selection (mutual_info) computed on training data only")
    logger.info(f"  ✓ Scaling (StandardScaler) fit on training data only")
    logger.info(f"  ✓ All operations performed inside CV folds\n")

    # Run hyperparameter search
    logger.info(f"{'=' * 80}")
    logger.info("STARTING HYPERPARAMETER SEARCH")
    logger.info(f"{'=' * 80}\n")

    all_results = []

    for combo_idx, (n_feat, k, w, m) in enumerate(combinations, 1):
        logger.info(f"[{combo_idx}/{n_combinations}] n_features={n_feat}, k={k}, weights={w}")

        result = evaluate_hyperparameter_combination(X, y, groups, n_feat, k, w, m)
        all_results.append(result)

        logger.info(
            f"  → Accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}  |  "
            f"Sensitivity: {result['mean_sensitivity']:.4f} ± {result['std_sensitivity']:.4f}  |  "
            f"AUC: {result['mean_auc']:.4f}"
        )

    # Convert results to DataFrame
    logger.info(f"\n{'=' * 80}")
    logger.info("ANALYZING RESULTS")
    logger.info(f"{'=' * 80}\n")

    results_df = pd.DataFrame(all_results)

    # Rank configurations
    ranked_results = rank_configurations(results_df)

    logger.info(f"Top 5 Configurations (ranked by sensitivity, AUC, accuracy, stability):\n")
    for idx, row in ranked_results.head(5).iterrows():
        logger.info(
            f"  #{row['rank']}. K={int(row['k'])}, Features={int(row['n_features'])}, "
            f"Weights={row['weights']}"
        )
        logger.info(
            f"     Sensitivity: {row['mean_sensitivity']:.4f} ± {row['std_sensitivity']:.4f}  |  "
            f"Accuracy: {row['mean_accuracy']:.4f}  |  AUC: {row['mean_auc']:.4f}"
        )
        logger.info("")

    # Get best configuration
    best_config = ranked_results.iloc[0]
    logger.info(f"{'=' * 80}")
    logger.info("BEST CONFIGURATION")
    logger.info(f"{'=' * 80}")
    logger.info(f"K (neighbors): {int(best_config['k'])}")
    logger.info(f"Features: {int(best_config['n_features'])}")
    logger.info(f"Weights: {best_config['weights']}")
    logger.info(f"Metric: {best_config['metric']}")
    logger.info(f"\nPerformance:")
    logger.info(f"  Mean Accuracy: {best_config['mean_accuracy']:.4f} ± {best_config['std_accuracy']:.4f}")
    logger.info(f"  Mean Sensitivity: {best_config['mean_sensitivity']:.4f} ± {best_config['std_sensitivity']:.4f}")
    logger.info(f"  Mean Specificity: {best_config['mean_specificity']:.4f}")
    logger.info(f"  Mean Precision: {best_config['mean_precision']:.4f}")
    logger.info(f"  Mean F1 Score: {best_config['mean_f1']:.4f}")
    logger.info(f"  Mean ROC-AUC: {best_config['mean_auc']:.4f}")

    # Save results to CSV
    logger.info(f"\n{'=' * 80}")
    logger.info("SAVING RESULTS")
    logger.info(f"{'=' * 80}\n")

    csv_path = OUT_DIR / "hyperparameter_search_results.csv"
    ranked_results_sorted = ranked_results.sort_values("rank")
    ranked_results_sorted.to_csv(csv_path, index=False)
    logger.info(f"✓ Saved ranked results: {csv_path}")

    # Save full results JSON
    json_path = OUT_DIR / "hyperparameter_search_full_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"✓ Saved full results JSON: {json_path}")

    # Generate visualizations
    logger.info(f"\nGenerating visualizations...")
    plot_hyperparameter_heatmaps(results_df)

    # Generate summary report
    logger.info(f"Generating summary report...")

    report_path = OUT_DIR / "HYPERPARAMETER_SEARCH_REPORT.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("KNN HYPERPARAMETER SEARCH - FINAL REPORT\n")
        f.write("Target + Novelty Concatenated Preprocessing\n")
        f.write("=" * 80 + "\n\n")

        f.write("DATASET SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Feature mode: {feature_mode}\n")
        f.write(f"Total epochs: {X.shape[0]}\n")
        f.write(f"Features (input): {X.shape[1]}\n")
        f.write(f"Total subjects: {len(np.unique(groups))}\n")
        f.write(f"CTL samples: {n_ctl}\n")
        f.write(f"PD samples: {n_pd}\n")
        f.write(f"Class balance: {n_pd / len(y):.2%} PD\n\n")

        f.write("SEARCH CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Algorithm: K-Nearest Neighbors\n")
        f.write(f"K values tested: {HYPERPARAMETER_GRID['k']}\n")
        f.write(f"Feature counts tested: {HYPERPARAMETER_GRID['n_features']}\n")
        f.write(f"Weights tested: {HYPERPARAMETER_GRID['weights']}\n")
        f.write(f"Distance metric: {HYPERPARAMETER_GRID['metric']} (fixed)\n")
        f.write(f"Cross-validation: {N_SPLITS}-Fold StratifiedGroupKFold (subject-aware)\n")
        f.write(f"Total combinations evaluated: {n_combinations}\n\n")

        f.write("TRAINING PIPELINE (Leakage Prevention)\n")
        f.write("-" * 80 + "\n")
        f.write("CRITICAL: Feature selection and scaling are performed INSIDE each CV fold\n")
        f.write("to prevent data leakage and ensure unbiased performance estimates.\n\n")
        f.write("Inside each fold, for each hyperparameter combination:\n")
        f.write("✓ Mutual Information feature selection → top N features (on training data only)\n")
        f.write("✓ StandardScaler normalization (fit on training data only)\n")
        f.write("✓ KNN training on selected + scaled features\n")
        f.write("✓ Validation data transformed using training-fitted selector/scaler\n\n")

        f.write("RANKING PRIORITY\n")
        f.write("-" * 80 + "\n")
        f.write("Configurations ranked by:\n")
        f.write("  1. Highest mean sensitivity (recall for PD) - PRIMARY\n")
        f.write("  2. Good ROC-AUC score\n")
        f.write("  3. Good overall accuracy\n")
        f.write("  4. Lower standard deviation (more stable across folds)\n\n")

        f.write("TOP 10 CONFIGURATIONS\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Rank':<6} {'K':<6} {'Features':<10} {'Weights':<12} "
            f"{'Accuracy':<12} {'Sensitivity':<12} {'Specificity':<12} {'AUC':<12}\n"
        )
        f.write("-" * 80 + "\n")

        for idx, row in ranked_results.head(10).iterrows():
            f.write(
                f"{int(row['rank']):<6} {int(row['k']):<6} {int(row['n_features']):<10} "
                f"{row['weights']:<12} {row['mean_accuracy']:<12.4f} "
                f"{row['mean_sensitivity']:<12.4f} {row['mean_specificity']:<12.4f} "
                f"{row['mean_auc']:<12.4f}\n"
            )
        f.write("\n")

        f.write("BEST CONFIGURATION (RECOMMENDED)\n")
        f.write("-" * 80 + "\n")
        f.write(f"K (neighbors): {int(best_config['k'])}\n")
        f.write(f"Features: {int(best_config['n_features'])}\n")
        f.write(f"Weights: {best_config['weights']}\n")
        f.write(f"Metric: {best_config['metric']}\n\n")
        f.write(f"Mean Accuracy:       {best_config['mean_accuracy']:.4f} ± {best_config['std_accuracy']:.4f}\n")
        f.write(f"Mean Sensitivity:    {best_config['mean_sensitivity']:.4f} ± {best_config['std_sensitivity']:.4f}\n")
        f.write(f"Mean Specificity:    {best_config['mean_specificity']:.4f}\n")
        f.write(f"Mean Precision:      {best_config['mean_precision']:.4f}\n")
        f.write(f"Mean F1 Score:       {best_config['mean_f1']:.4f}\n")
        f.write(f"Mean ROC-AUC:        {best_config['mean_auc']:.4f} ± {best_config['std_auc']:.4f}\n\n")

        f.write("INTERPRETATION\n")
        f.write("-" * 80 + "\n")

        # Stability analysis
        avg_std_sens = results_df["std_sensitivity"].mean()
        best_std_sens = best_config["std_sensitivity"]

        if best_std_sens < avg_std_sens * 0.8:
            stability_comment = (
                f"The best configuration shows GOOD STABILITY with sensitivity "
                f"std dev of {best_std_sens:.4f} (below average of {avg_std_sens:.4f}). "
                f"Model is relatively consistent across folds."
            )
        elif best_std_sens > avg_std_sens * 1.2:
            stability_comment = (
                f"The best configuration shows VARIABLE PERFORMANCE with sensitivity "
                f"std dev of {best_std_sens:.4f} (above average of {avg_std_sens:.4f}). "
                f"Model performance varies across folds - consider using ensemble or "
                f"monitoring performance carefully."
            )
        else:
            stability_comment = (
                f"The best configuration shows MODERATE STABILITY with sensitivity "
                f"std dev of {best_std_sens:.4f} (near average of {avg_std_sens:.4f})."
            )

        f.write(stability_comment + "\n\n")

        f.write("Sensitivity analysis:\n")
        f.write(f"  The best configuration achieves {best_config['mean_sensitivity']:.2%} sensitivity,\n")
        f.write(f"  meaning it identifies approximately {best_config['mean_sensitivity']:.0%} of PD cases correctly.\n")
        f.write(f"  With {best_config['mean_specificity']:.2%} specificity, it correctly identifies\n")
        f.write(f"  {best_config['mean_specificity']:.0%} of control subjects.\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    logger.info(f"✓ Saved summary report: {report_path}")

    logger.info(f"\n{'=' * 80}")
    logger.info("✅ HYPERPARAMETER SEARCH COMPLETE!")
    logger.info(f"{'=' * 80}")
    logger.info(f"\nOutput directory: {OUT_DIR}")
    logger.info(f"  • Results CSV: {csv_path}")
    logger.info(f"  • Full results JSON: {json_path}")
    logger.info(f"  • Summary report: {report_path}")
    logger.info(f"  • Visualizations: {PLOTS_DIR}")
    logger.info(f"\nBest configuration:")
    logger.info(f"  K={int(best_config['k'])}, Features={int(best_config['n_features'])}, Weights={best_config['weights']}")
    logger.info(f"  Sensitivity: {best_config['mean_sensitivity']:.4f}  |  AUC: {best_config['mean_auc']:.4f}")


if __name__ == "__main__":
    main()
