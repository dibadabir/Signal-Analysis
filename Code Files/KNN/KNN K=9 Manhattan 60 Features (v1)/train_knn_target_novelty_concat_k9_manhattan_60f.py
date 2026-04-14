# ============================================================
# train_knn_target_novelty_concat_k9_manhattan_60f.py
# ============================================================
# KNN Training on Target + Novelty Concatenated Preprocessing
# Version: Conservative hyperparameters (K=9, Manhattan, 60 features)
# 5-Fold StratifiedGroupKFold Cross-Validation
# CRITICAL: Feature selection and scaling happen INSIDE CV folds
#           to prevent data leakage and ensure unbiased estimates
# ============================================================

import json
import logging
from pathlib import Path
from typing import List, Tuple

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
    roc_auc_score, roc_curve, confusion_matrix, log_loss
)
from sklearn.decomposition import PCA

# ============================================================
# Configuration
# ============================================================
FEATURES_PATH = Path("outputs") / "FEATURES_KNN_TARGET_NOVELTY_CONCAT.npz"
OUT_DIR = Path("outputs") / "knn_version_target_novelty_k9_manhattan_60f"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# KNN Configuration (CONSERVATIVE: K=9, Manhattan, Uniform weights)
# ============================================================
DEFAULT_KNN_K = 9
DEFAULT_KNN_METRIC = "manhattan"
DEFAULT_KNN_WEIGHTS = "uniform"

# Feature Selection Configuration
# CRITICAL: Feature selection happens INSIDE each CV fold to avoid leakage
N_SELECTED_FEATURES = 60

# Cross-validation
N_SPLITS = 5

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_knn_target_novelty_k9")


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
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)

    # Safely compute ROC AUC and log loss
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            metrics["log_loss"] = log_loss(y_true, y_proba)
        except Exception:
            metrics["roc_auc"] = np.nan
            metrics["log_loss"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
        metrics["log_loss"] = np.nan

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


def plot_all_confusion_matrices(all_cms: List, fold_accs: List, n_splits: int):
    """Create a single figure with all fold confusion matrices side-by-side."""
    fig, axes = plt.subplots(1, n_splits, figsize=(20, 4))
    fig.suptitle("Confusion Matrices - All 5 Folds", fontsize=16, fontweight="bold", y=1.02)

    for fold_idx in range(n_splits):
        cm = all_cms[fold_idx]
        acc = fold_accs[fold_idx]

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=axes[fold_idx],
            cbar=False, square=True, annot_kws={"fontsize": 12}
        )
        axes[fold_idx].set_title(f"Fold {fold_idx + 1}\nAccuracy: {acc:.2%}", fontsize=12, fontweight="bold")
        axes[fold_idx].set_ylabel("True Label" if fold_idx == 0 else "", fontsize=10)
        axes[fold_idx].set_xlabel("Predicted Label", fontsize=10)
        axes[fold_idx].set_xticklabels(["CTL", "PD"], rotation=0)
        axes[fold_idx].set_yticklabels(["CTL", "PD"], rotation=90)

    plt.tight_layout()
    cm_path = PLOTS_DIR / "all_folds_confusion_matrices.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    logger.info(f"  ✓ Saved confusion matrices: {cm_path}")
    plt.close()


def plot_feature_space_pca(X_train, y_train, X_val, y_val, fold_idx: int):
    """Plot 2D feature scatter (PCA-projected) with separate train/validation."""
    try:
        # Project to 2D using PCA
        pca = PCA(n_components=2)
        X_train_2d = pca.fit_transform(X_train)
        X_val_2d = pca.transform(X_val)

        # Create subplots - simple scatter plot without mesh
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Feature Space (PCA) - Fold {fold_idx + 1}", fontsize=14, fontweight="bold")

        for ax, X_2d, y, title in [
            (axes[0], X_train_2d, y_train, "Training Data"),
            (axes[1], X_val_2d, y_val, "Validation Data"),
        ]:
            # Plot data points only
            ax.scatter(
                X_2d[y == 0, 0], X_2d[y == 0, 1],
                c="blue", label="CTL", marker="o", s=40, alpha=0.6, edgecolors="navy"
            )
            ax.scatter(
                X_2d[y == 1, 0], X_2d[y == 1, 1],
                c="red", label="PD", marker="^", s=40, alpha=0.6, edgecolors="darkred"
            )

            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", fontsize=11)
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)", fontsize=11)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        boundary_path = PLOTS_DIR / f"feature_space_fold{fold_idx + 1}.png"
        plt.savefig(boundary_path, dpi=150, bbox_inches="tight")
        logger.info(f"  ✓ Saved feature space plot (Fold {fold_idx + 1}): {boundary_path}")
        plt.close()
    except Exception as e:
        logger.warning(f"  ⚠️  Could not generate visualization: {e}")


def plot_roc_curves(all_roc_data: List[Tuple], n_splits: int):
    """Plot ROC curves for all folds."""
    try:
        fig, axes = plt.subplots(1, n_splits + 1, figsize=(22, 4))
        fig.suptitle("ROC Curves - All 5 Folds + Mean", fontsize=16, fontweight="bold", y=1.02)

        all_fpr = []
        all_tpr = []

        for fold_idx, (fpr, tpr, auc) in enumerate(all_roc_data):
            all_fpr.append(fpr)
            all_tpr.append(tpr)

            axes[fold_idx].plot(fpr, tpr, color="red", lw=2, label=f"AUC = {auc:.3f}")
            axes[fold_idx].plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
            axes[fold_idx].set_xlim([0.0, 1.0])
            axes[fold_idx].set_ylim([0.0, 1.05])
            axes[fold_idx].set_xlabel("False Positive Rate", fontsize=10)
            axes[fold_idx].set_ylabel("True Positive Rate" if fold_idx == 0 else "", fontsize=10)
            axes[fold_idx].set_title(f"Fold {fold_idx + 1}", fontsize=12, fontweight="bold")
            axes[fold_idx].legend(fontsize=9)
            axes[fold_idx].grid(True, alpha=0.3)

        # Mean ROC curve
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)], axis=0)
        mean_auc = np.mean([auc for _, _, auc in all_roc_data])

        axes[-1].plot(mean_fpr, mean_tpr, color="red", lw=2, label=f"Mean AUC = {mean_auc:.3f}")
        axes[-1].plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
        axes[-1].set_xlim([0.0, 1.0])
        axes[-1].set_ylim([0.0, 1.05])
        axes[-1].set_xlabel("False Positive Rate", fontsize=10)
        axes[-1].set_title("Mean ROC", fontsize=12, fontweight="bold")
        axes[-1].legend(fontsize=9)
        axes[-1].grid(True, alpha=0.3)

        plt.tight_layout()
        roc_path = PLOTS_DIR / "roc_curves_all_folds.png"
        plt.savefig(roc_path, dpi=150, bbox_inches="tight")
        logger.info(f"  ✓ Saved ROC curves: {roc_path}")
        plt.close()
    except Exception as e:
        logger.warning(f"  ⚠️  Could not generate ROC curves: {e}")


# ============================================================
# Main Training Function
# ============================================================

def main():
    logger.info("=" * 75)
    logger.info("KNN TRAINING — TARGET + NOVELTY CONCATENATED PREPROCESSING")
    logger.info("Version: Conservative (K=9, Manhattan, 60 features)")
    logger.info("=" * 75)

    # Load features
    logger.info(f"\nLoading features from: {FEATURES_PATH}")
    data = np.load(FEATURES_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    
    # Safely extract feature mode from npz metadata
    if "feature_mode" in data.files:
        feature_mode = str(data["feature_mode"])
    else:
        feature_mode = "unknown"
        logger.warning("  ⚠️  Feature mode not found in npz file, using default")

    logger.info(f"  Feature mode: {feature_mode}")
    logger.info(f"  Total epochs: {X.shape[0]}")
    logger.info(f"  Input features: {X.shape[1]}")
    logger.info(f"  Classes: {np.unique(y, return_counts=True)}")
    logger.info(f"  Subjects: {len(np.unique(groups))}")
    
    n_pd = np.sum(y == 1)
    n_ctl = np.sum(y == 0)
    logger.info(f"  Class balance: {n_pd} PD, {n_ctl} CTL ({n_pd/len(y):.1%} PD)")

    # ========================================================================
    # TRAINING WITH FEATURE SELECTION & SCALING INSIDE CV FOLDS
    # ========================================================================
    logger.info(f"\n{'='*75}")
    logger.info("TRAINING WITH IN-FOLD FEATURE SELECTION & SCALING")
    logger.info("(Prevents data leakage and ensures unbiased estimates)")
    logger.info(f"{'='*75}\n")
    
    # Use conservative parameters for training
    KNN_K = DEFAULT_KNN_K
    KNN_METRIC = DEFAULT_KNN_METRIC
    KNN_WEIGHTS = DEFAULT_KNN_WEIGHTS
    N_FEATURES = N_SELECTED_FEATURES

    logger.info(f"Configuration:")
    logger.info(f"  KNN: K={KNN_K}, metric={KNN_METRIC}, weights={KNN_WEIGHTS}")
    logger.info(f"  Feature selection: {N_FEATURES} features (mutual information)")
    logger.info(f"  Scaling: StandardScaler")
    logger.info(f"  Cross-validation: {N_SPLITS}-Fold StratifiedGroupKFold")
    logger.info(f"  Data leakage prevention: Feature selection & scaling inside folds only\n")

    all_metrics = []
    all_cms = []
    all_fold_accs = []
    all_roc_data = []
    all_fold_results = {}
    fold_idx = 0

    # Cross-validation setup
    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # Cross-validation loop
    for train_idx, val_idx in skf.split(X, y, groups):
        fold_idx += 1
        logger.info(f"\n{'='*75}")
        logger.info(f"FOLD {fold_idx}/{N_SPLITS}")
        logger.info(f"{'='*75}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        n_train_pd = np.sum(y_train == 1)
        n_train_ctl = np.sum(y_train == 0)
        n_val_pd = np.sum(y_val == 1)
        n_val_ctl = np.sum(y_val == 0)

        logger.info(f"\nData split:")
        logger.info(f"  Train: {len(y_train)} epochs ({n_train_pd} PD, {n_train_ctl} CTL) - {n_train_pd/len(y_train):.1%} PD")
        logger.info(f"  Val:   {len(y_val)} epochs ({n_val_pd} PD, {n_val_ctl} CTL) - {n_val_pd/len(y_val):.1%} PD")
        logger.info(f"  Train subjects: {len(np.unique(groups[train_idx]))}")
        logger.info(f"  Val subjects: {len(np.unique(groups[val_idx]))}")

        # ═════════════════════════════════════════════════════════════════
        # STEP 1: Feature Selection (INSIDE fold - using training data only)
        # ═════════════════════════════════════════════════════════════════
        logger.info(f"\n[1/3] Feature Selection:")
        logger.info(f"  Computing mutual information on training data only...")
        
        X_train_sel, X_val_sel, sel_indices, mi_scores = select_features_inside_fold(
            X_train, y_train, X_val, n_features=N_FEATURES
        )
        
        logger.info(f"  ✓ Selected {len(sel_indices)} features")
        logger.info(f"  MI scores range: {mi_scores.min():.6f} to {mi_scores.max():.6f}")

        # ═════════════════════════════════════════════════════════════════
        # STEP 2: Feature Scaling (INSIDE fold - using training data only)
        # ═════════════════════════════════════════════════════════════════
        logger.info(f"\n[2/3] Feature Scaling (StandardScaler):")
        logger.info(f"  Fitting scaler on training data only...")
        
        X_train_scaled, X_val_scaled, scaler = scale_inside_fold(X_train_sel, X_val_sel)
        
        logger.info(f"  ✓ Scaler fitted on training data")
        logger.info(f"  Train mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}")
        logger.info(f"  Val mean: {X_val_scaled.mean():.6f}, std: {X_val_scaled.std():.6f}")

        # ═════════════════════════════════════════════════════════════════
        # STEP 3: KNN Training (INSIDE fold - on scaled selected features)
        # ═════════════════════════════════════════════════════════════════
        logger.info(f"\n[3/3] KNN Training:")
        
        model = KNeighborsClassifier(
            n_neighbors=KNN_K,
            metric=KNN_METRIC,
            weights=KNN_WEIGHTS,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        logger.info(f"  ✓ KNN model trained (K={KNN_K}, metric={KNN_METRIC}, weights={KNN_WEIGHTS})")

        # ═════════════════════════════════════════════════════════════════
        # Predictions and Evaluation
        # ═════════════════════════════════════════════════════════════════
        logger.info(f"\nEvaluation:")
        
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
        y_val_proba = model.predict_proba(X_val_scaled)[:, 1]

        # Calculate metrics
        train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
        val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)

        logger.info(f"\n  Training metrics:")
        logger.info(f"    Accuracy:  {train_metrics['accuracy']:.4f}")
        logger.info(f"    Precision: {train_metrics['precision']:.4f}")
        logger.info(f"    Sensitivity: {train_metrics['recall']:.4f}")
        logger.info(f"    F1 Score:  {train_metrics['f1_score']:.4f}")
        logger.info(f"    ROC-AUC:   {train_metrics.get('roc_auc', 0):.4f}")

        logger.info(f"\n  Validation metrics:")
        logger.info(f"    Accuracy:  {val_metrics['accuracy']:.4f}")
        logger.info(f"    Precision: {val_metrics['precision']:.4f}")
        logger.info(f"    Sensitivity: {val_metrics['sensitivity']:.4f}")
        logger.info(f"    Specificity: {val_metrics['specificity']:.4f}")
        logger.info(f"    FNR:       {val_metrics['false_negative_rate']:.4f}")
        logger.info(f"    F1 Score:  {val_metrics['f1_score']:.4f}")
        logger.info(f"    ROC-AUC:   {val_metrics.get('roc_auc', 0):.4f}")

        # Store results (use labels=[0, 1] for robustness)
        cm = confusion_matrix(y_val, y_val_pred, labels=[0, 1])
        all_cms.append(cm)
        all_fold_accs.append(val_metrics["accuracy"])

        # Safely compute ROC curve (skip if only one class present)
        try:
            if len(np.unique(y_val)) > 1:
                fpr, tpr, _ = roc_curve(y_val, y_val_proba)
                auc = val_metrics.get("roc_auc", 0)
            else:
                logger.warning(f"    ⚠️  Only one class in validation set, skipping ROC curve")
                fpr, tpr, auc = np.array([0, 1]), np.array([0, 1]), np.nan
        except Exception as e:
            logger.warning(f"    ⚠️  Could not compute ROC curve: {e}")
            fpr, tpr, auc = np.array([0, 1]), np.array([0, 1]), np.nan
        all_roc_data.append((fpr, tpr, auc))

        all_metrics.append({
            "fold": fold_idx,
            "n_train_samples": len(y_train),
            "n_val_samples": len(y_val),
            "n_train_subjects": len(np.unique(groups[train_idx])),
            "n_val_subjects": len(np.unique(groups[val_idx])),
            "n_features_input": X.shape[1],
            "n_features_selected": N_FEATURES,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_f1": train_metrics["f1_score"],
            "val_f1": val_metrics["f1_score"],
            "train_precision": train_metrics["precision"],
            "val_precision": val_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "val_recall": val_metrics["recall"],
            "train_roc_auc": train_metrics.get("roc_auc", 0),
            "val_roc_auc": val_metrics.get("roc_auc", 0),
            "val_specificity": val_metrics["specificity"],
            "val_fnr": val_metrics["false_negative_rate"],
        })

        all_fold_results[f"fold_{fold_idx}"] = {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "selected_feature_indices": sel_indices.tolist(),
            "mi_scores": mi_scores.tolist(),
        }

        # Generate fold-specific visualizations
        logger.info(f"\nGenerating visualizations...")
        plot_feature_space_pca(X_train_scaled, y_train, X_val_scaled, y_val, fold_idx - 1)

    # Generate combined visualizations
    logger.info(f"\nGenerating combined visualizations...")
    plot_all_confusion_matrices(all_cms, all_fold_accs, N_SPLITS)
    plot_roc_curves(all_roc_data, N_SPLITS)

    # Calculate mean metrics
    logger.info(f"\n{'=' * 75}")
    logger.info("CROSS-VALIDATION RESULTS (Mean across all folds)")
    logger.info(f"{'=' * 75}\n")

    metrics_df = pd.DataFrame(all_metrics)

    mean_val_acc = metrics_df["val_accuracy"].mean()
    std_val_acc = metrics_df["val_accuracy"].std()

    mean_val_f1 = metrics_df["val_f1"].mean()
    std_val_f1 = metrics_df["val_f1"].std()

    mean_val_prec = metrics_df["val_precision"].mean()
    std_val_prec = metrics_df["val_precision"].std()

    mean_val_rec = metrics_df["val_recall"].mean()
    std_val_rec = metrics_df["val_recall"].std()

    mean_val_auc = np.nanmean(metrics_df["val_roc_auc"])
    std_val_auc = np.nanstd(metrics_df["val_roc_auc"])

    # Compute mean specificity and FNR from stored metrics (more robust)
    mean_val_spec = metrics_df["val_specificity"].mean()
    mean_val_fnr = metrics_df["val_fnr"].mean()

    logger.info(f"Accuracy:       {mean_val_acc:.4f} ± {std_val_acc:.4f}")
    logger.info(f"Sensitivity:    {mean_val_rec:.4f} ± {std_val_rec:.4f}")
    logger.info(f"Specificity:    {mean_val_spec:.4f}")
    logger.info(f"Precision:      {mean_val_prec:.4f} ± {std_val_prec:.4f}")
    logger.info(f"F1 Score:       {mean_val_f1:.4f} ± {std_val_f1:.4f}")
    logger.info(f"False Neg Rate: {mean_val_fnr:.4f}")
    logger.info(f"ROC-AUC:        {mean_val_auc:.4f} ± {std_val_auc:.4f}")

    # Save metrics to CSV
    metrics_csv = OUT_DIR / "cv_fold_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    logger.info(f"\n✓ Saved fold metrics: {metrics_csv}")

    # Generate final report
    logger.info(f"\nGenerating final report...")

    report_path = OUT_DIR / "FINAL_REPORT_KNN_TARGET_NOVELTY_K9_MANHATTAN.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("KNN CLASSIFICATION MODEL - FINAL REPORT\n")
        f.write("Target + Novelty Concatenated Preprocessing\n")
        f.write("Version: Conservative (K=9, Manhattan, 60 features)\n")
        f.write("=" * 80 + "\n\n")

        f.write("DATASET SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Feature mode: {feature_mode}\n")
        f.write(f"Total epochs: {X.shape[0]}\n")
        f.write(f"Features (input): {X.shape[1]}\n")
        f.write(f"  → Target features: {X.shape[1]//2}\n")
        f.write(f"  → Novelty features: {X.shape[1]//2}\n")
        f.write(f"Total subjects: {len(np.unique(groups))}\n")
        f.write(f"CTL samples: {np.sum(y == 0)}\n")
        f.write(f"PD samples: {np.sum(y == 1)}\n")
        f.write(f"Class balance: {np.sum(y == 1) / len(y):.2%} PD\n\n")

        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Algorithm: K-Nearest Neighbors\n")
        f.write(f"K value: {KNN_K}\n")
        f.write(f"Distance metric: {KNN_METRIC}\n")
        f.write(f"Weights: {KNN_WEIGHTS}\n")
        f.write(f"Cross-validation: {N_SPLITS}-Fold StratifiedGroupKFold\n\n")

        f.write("PREPROCESSING PIPELINE\n")
        f.write("-" * 80 + "\n")
        f.write("✓ Bad channel detection & interpolation\n")
        f.write("✓ Average re-referencing\n")
        f.write("✓ Notch filtering (60 Hz + harmonics)\n")
        f.write("✓ Bandpass filtering (0.5-40 Hz)\n")
        f.write("✓ ICA artifact removal (EOG components)\n")
        f.write("✓ Channel selection (59 standard 10-20)\n")
        f.write("✓ Event-locked epoching (-250 to +1250 ms)\n")
        f.write("✓ Peak-to-peak artifact rejection (>150 µV)\n")
        f.write("✓ Multi-stage artifact detection (PTP, kurtosis, muscle)\n")
        f.write("✓ 1,392 Welch PSD features per condition\n")
        f.write("✓ Target + Novelty concatenated (2,784 total features)\n")
        f.write("✓ Trial matching (balanced target/novelty counts)\n\n")

        f.write("TRAINING PIPELINE (Leakage Prevention)\n")
        f.write("-" * 80 + "\n")
        f.write("CRITICAL: Feature selection and scaling are performed INSIDE each CV fold\n")
        f.write("to prevent data leakage and ensure unbiased performance estimates.\n\n")
        f.write("Inside each fold:\n")
        f.write(f"✓ Mutual Information feature selection → {N_FEATURES} features (on training data only)\n")
        f.write("✓ StandardScaler normalization (fit on training data only)\n")
        f.write("✓ KNN training on selected + scaled features\n")
        f.write("✓ Validation data transformed using training-fitted selector/scaler\n\n")

        f.write("PER-FOLD PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Fold':<6} {'Train Acc':<12} {'Val Acc':<12} {'Val Sens':<12} {'Val Spec':<12} {'Val AUC':<12}\n")
        f.write("-" * 80 + "\n")
        for idx, row in metrics_df.iterrows():
            val_sens = all_fold_results[f"fold_{int(row['fold'])}"]["val_metrics"]["sensitivity"]
            val_spec = all_fold_results[f"fold_{int(row['fold'])}"]["val_metrics"]["specificity"]
            f.write(f"{int(row['fold']):<6} {row['train_accuracy']:<12.4f} {row['val_accuracy']:<12.4f} {val_sens:<12.4f} {val_spec:<12.4f} {row['val_roc_auc']:<12.4f}\n")
        f.write("\n")

        f.write("MEAN PERFORMANCE METRICS (5-FOLD CV WITH IN-FOLD FEATURE SELECTION)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:           {mean_val_acc:.4f} ± {std_val_acc:.4f}\n")
        f.write(f"Sensitivity (Recall): {mean_val_rec:.4f} ± {std_val_rec:.4f}\n")
        f.write(f"Specificity:        {mean_val_spec:.4f}\n")
        f.write(f"Precision:          {mean_val_prec:.4f} ± {std_val_prec:.4f}\n")
        f.write(f"F1 Score:           {mean_val_f1:.4f} ± {std_val_f1:.4f}\n")
        f.write(f"False Negative Rate: {mean_val_fnr:.4f}\n")
        f.write(f"ROC-AUC:            {mean_val_auc:.4f} ± {std_val_auc:.4f}\n\n")

        f.write("DETAILED FOLD-BY-FOLD RESULTS\n")
        f.write("-" * 80 + "\n\n")

        for fold_num in range(1, N_SPLITS + 1):
            fold_key = f"fold_{fold_num}"
            train_m = all_fold_results[fold_key]["train_metrics"]
            val_m = all_fold_results[fold_key]["val_metrics"]
            row = metrics_df[metrics_df["fold"] == fold_num].iloc[0]

            f.write(f"FOLD {fold_num}\n")
            f.write("~" * 80 + "\n")
            f.write(f"Sample sizes: Train={int(row['n_train_samples'])}, Val={int(row['n_val_samples'])}\n")
            f.write(f"Subject counts: Train={int(row['n_train_subjects'])}, Val={int(row['n_val_subjects'])}\n")
            f.write(f"Features: Input={int(row['n_features_input'])}, Selected={int(row['n_features_selected'])}\n\n")
            f.write(f"Train Accuracy:         {train_m['accuracy']:.4f}  |  Val Accuracy:        {val_m['accuracy']:.4f}\n")
            f.write(f"Train Precision:        {train_m['precision']:.4f}  |  Val Precision:       {val_m['precision']:.4f}\n")
            f.write(f"Train Recall:           {train_m['recall']:.4f}  |  Val Recall:          {val_m['recall']:.4f}\n")
            f.write(f"Train F1 Score:         {train_m['f1_score']:.4f}  |  Val F1 Score:        {val_m['f1_score']:.4f}\n")
            f.write(f"Train ROC-AUC:          {train_m.get('roc_auc', 0):.4f}  |  Val ROC-AUC:         {val_m.get('roc_auc', 0):.4f}\n")
            f.write(f"\nSensitivity (Recall):   {val_m['sensitivity']:.4f}  (True Positive Rate)\n")
            f.write(f"Specificity:            {val_m['specificity']:.4f}  (True Negative Rate)\n")
            f.write(f"False Negative Rate:    {val_m['false_negative_rate']:.4f}  (Critical for PD diagnosis)\n")
            f.write(f"\nConfusion Matrix:\n")
            f.write(f"  TN={val_m['true_negatives']:<3}  FP={val_m['false_positives']:<3}\n")
            f.write(f"  FN={val_m['false_negatives']:<3}  TP={val_m['true_positives']:<3}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("SUMMARY & INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"The conservative KNN model (K=9, Manhattan, 60 features) achieved a mean\n")
        f.write(f"accuracy of {mean_val_acc:.2%} across 5-fold cross-validation, with ROC-AUC\n")
        f.write(f"of {mean_val_auc:.3f}, using target + novelty concatenated preprocessing.\n\n")
        f.write(f"Key findings:\n")
        f.write(f"• Sensitivity (recall): {mean_val_rec:.2%} - captures {mean_val_rec:.0%} of PD cases\n")
        f.write(f"• Specificity: {mean_val_spec:.2%} - correctly identifies {mean_val_spec:.0%} of controls\n")
        f.write(f"• False Negative Rate: {mean_val_fnr:.2%} - misses {mean_val_fnr:.0%} of actual PD cases\n")
        f.write(f"• F1 Score: {mean_val_f1:.4f} - balanced precision-recall performance\n\n")
        f.write(f"Data leakage prevention:\n")
        f.write(f"• Feature selection (mutual information) is performed independently within\n")
        f.write(f"  each CV fold using training data only\n")
        f.write(f"• Scaling (StandardScaler) is fit on training data only and applied to\n")
        f.write(f"  validation data using training statistics\n")
        f.write(f"• This ensures unbiased performance estimates and prevents information\n")
        f.write(f"  about validation samples from influencing the model\n\n")
        f.write("=" * 80 + "\n")

    logger.info(f"✓ Final report saved: {report_path}")

    # Save full results JSON
    results_json = OUT_DIR / "model_results.json"
    with open(results_json, "w") as f:
        json.dump(all_fold_results, f, indent=2, default=str)
    logger.info(f"✓ Saved detailed results: {results_json}")

    # Save configuration
    config_json = OUT_DIR / "config.json"
    config = {
        "preprocessing_file": str(FEATURES_PATH),
        "feature_mode": feature_mode,
        "n_input_features": int(X.shape[1]),
        "n_selected_features": N_FEATURES,
        "knn_k": KNN_K,
        "knn_metric": KNN_METRIC,
        "knn_weights": KNN_WEIGHTS,
        "n_splits": N_SPLITS,
        "random_state": 42,
        "feature_selection": "mutual_info_classif",
        "scaling": "StandardScaler",
        "cv_strategy": "StratifiedGroupKFold",
        "version": "conservative_k9_manhattan_60f",
    }
    with open(config_json, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"✓ Saved configuration: {config_json}")

    logger.info(f"\n{'=' * 75}")
    logger.info("✅ TRAINING COMPLETE!")
    logger.info(f"{'=' * 75}")
    logger.info(f"\nOutput directory: {OUT_DIR}")
    logger.info(f"  • Visualizations: {PLOTS_DIR}")
    logger.info(f"  • Final report: {report_path}")
    logger.info(f"  • Metrics CSV: {metrics_csv}")
    logger.info(f"  • Results JSON: {results_json}")
    logger.info(f"  • Config: {config_json}")


if __name__ == "__main__":
    main()
