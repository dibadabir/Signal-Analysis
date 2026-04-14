# ============================================================
# train_knn_target_novelty_k9_manhattan_80f_holdout10.py
# ============================================================
# KNN Training on Target + Novelty Concatenated Preprocessing
# Version: Conservative hyperparameters (K=9, Manhattan, 80 features)
# 5-Fold StratifiedGroupKFold Cross-Validation on training pool
# 10-Subject Holdout Test Set (5 PD, 5 CTL) for final external validation
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
    roc_auc_score, roc_curve, confusion_matrix, log_loss
)
from sklearn.decomposition import PCA

# ============================================================
# Configuration
# ============================================================
FEATURES_PATH = Path("outputs") / "FEATURES_KNN_TARGET_NOVELTY_CONCAT.npz"
OUT_DIR = Path("outputs") / "knn_version_target_novelty_k9_manhattan_80f_holdout10"
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
N_SELECTED_FEATURES = 80

# Cross-validation
N_SPLITS = 5

# Holdout Configuration
N_HOLDOUT_SUBJECTS = 10
N_HOLDOUT_PD = 5
N_HOLDOUT_CTL = 5
HOLDOUT_RANDOM_STATE = 42

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_knn_target_novelty_k9_holdout10")


# ============================================================
# Helper Functions
# ============================================================

def select_holdout_subjects(groups, y, n_pd=5, n_ctl=5, random_state=42):
    """
    Select holdout subjects stratified by class (PD vs CTL).
    
    Returns:
    - holdout_subject_ids: list of subject IDs to hold out
    - holdout_mask: boolean mask (True = holdout, False = train)
    - subject_to_class: dict mapping subject ID to label
    """
    np.random.seed(random_state)
    
    # Get unique subjects and their classes
    unique_subjects = np.unique(groups)
    subject_to_class = {}
    
    for subj in unique_subjects:
        subj_mask = groups == subj
        subj_labels = y[subj_mask]
        # All epochs for a subject should have the same label
        subject_to_class[subj] = int(subj_labels[0])
    
    # Separate subjects by class
    pd_subjects = [subj for subj, label in subject_to_class.items() if label == 1]
    ctl_subjects = [subj for subj, label in subject_to_class.items() if label == 0]
    
    logger.info(f"\nAvailable subjects:")
    logger.info(f"  PD subjects: {len(pd_subjects)}")
    logger.info(f"  CTL subjects: {len(ctl_subjects)}")
    
    # Randomly select holdout subjects
    selected_pd = np.random.choice(pd_subjects, size=min(n_pd, len(pd_subjects)), replace=False)
    selected_ctl = np.random.choice(ctl_subjects, size=min(n_ctl, len(ctl_subjects)), replace=False)
    
    holdout_subject_ids = list(selected_pd) + list(selected_ctl)
    
    logger.info(f"\nHoldout subjects (random_state={random_state}):")
    logger.info(f"  PD: {sorted(selected_pd.tolist())}")
    logger.info(f"  CTL: {sorted(selected_ctl.tolist())}")
    logger.info(f"  Total: {len(holdout_subject_ids)} subjects")
    
    # Create boolean mask
    holdout_mask = np.isin(groups, holdout_subject_ids)
    
    return holdout_subject_ids, holdout_mask, subject_to_class


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


def select_features_inside_fold(X_train, y_train, X_val, n_features=80):
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
    fig.suptitle("Confusion Matrices - All 5 Folds (CV on Training Pool)", fontsize=16, fontweight="bold", y=1.02)

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
    cm_path = PLOTS_DIR / "cv_all_folds_confusion_matrices.png"
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
        boundary_path = PLOTS_DIR / f"cv_feature_space_fold{fold_idx + 1}.png"
        plt.savefig(boundary_path, dpi=150, bbox_inches="tight")
        logger.info(f"  ✓ Saved feature space plot (Fold {fold_idx + 1}): {boundary_path}")
        plt.close()
    except Exception as e:
        logger.warning(f"  ⚠️  Could not generate visualization: {e}")


def plot_roc_curves(all_roc_data: List[Tuple], n_splits: int):
    """Plot ROC curves for all folds."""
    try:
        fig, axes = plt.subplots(1, n_splits + 1, figsize=(22, 4))
        fig.suptitle("ROC Curves - All 5 Folds + Mean (CV on Training Pool)", fontsize=16, fontweight="bold", y=1.02)

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
        roc_path = PLOTS_DIR / "cv_roc_curves_all_folds.png"
        plt.savefig(roc_path, dpi=150, bbox_inches="tight")
        logger.info(f"  ✓ Saved ROC curves: {roc_path}")
        plt.close()
    except Exception as e:
        logger.warning(f"  ⚠️  Could not generate ROC curves: {e}")


def evaluate_holdout_subjects(X_holdout, y_holdout, groups_holdout, y_pred_holdout, 
                             y_proba_holdout: np.ndarray, subject_to_class: Dict) -> Dict:
    """
    Evaluate holdout performance at both epoch and subject levels.
    
    Returns:
    - holdout_results: dict with epoch-level and subject-level metrics
    """
    # Get unique holdout subjects
    holdout_subjects = np.unique(groups_holdout)
    
    results = {
        "epoch_level": {},
        "subject_level": {}
    }
    
    # Epoch-level metrics
    epoch_metrics = calculate_metrics(y_holdout, y_pred_holdout, y_proba_holdout)
    
    # Epoch-level confusion matrix
    epoch_cm = confusion_matrix(y_holdout, y_pred_holdout, labels=[0, 1])
    epoch_metrics["confusion_matrix"] = epoch_cm.tolist()
    
    results["epoch_level"] = epoch_metrics
    
    logger.info(f"\nHoldout Epoch-Level Evaluation:")
    logger.info(f"  Accuracy:  {epoch_metrics['accuracy']:.4f}")
    logger.info(f"  Sensitivity: {epoch_metrics['sensitivity']:.4f}")
    logger.info(f"  Specificity: {epoch_metrics['specificity']:.4f}")
    logger.info(f"  Precision: {epoch_metrics['precision']:.4f}")
    logger.info(f"  F1 Score:  {epoch_metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC:   {epoch_metrics.get('roc_auc', 0):.4f}")
    
    # Subject-level aggregation: Majority Vote
    logger.info(f"\nHoldout Subject-Level Evaluation (Majority Vote):")
    logger.info(f"{'Subject':<10} {'True Label':<15} {'PD Votes':<12} {'CTL Votes':<12} {'Predicted':<12} {'Correct':<10}")
    logger.info("-" * 80)
    
    subject_votes = {}
    subject_preds = {}
    subject_corrects = {}
    
    for subj in sorted(holdout_subjects):
        subj_mask = groups_holdout == subj
        subj_y_true = y_holdout[subj_mask]
        subj_y_pred = y_pred_holdout[subj_mask]
        
        # True label (all same for a subject)
        true_label = int(subj_y_true[0])
        
        # Count votes
        n_pd_votes = np.sum(subj_y_pred == 1)
        n_ctl_votes = np.sum(subj_y_pred == 0)
        
        # Majority vote prediction
        if n_pd_votes > n_ctl_votes:
            predicted = 1
        else:
            predicted = 0
        
        is_correct = (predicted == true_label)
        
        # Convert subj to Python int for JSON serialization
        subj_key = int(subj)
        subject_votes[subj_key] = {"pd_votes": int(n_pd_votes), "ctl_votes": int(n_ctl_votes)}
        subject_preds[subj_key] = int(predicted)
        subject_corrects[subj_key] = is_correct
        
        true_label_str = "PD" if true_label == 1 else "CTL"
        pred_label_str = "PD" if predicted == 1 else "CTL"
        correct_str = "✓" if is_correct else "✗"
        
        logger.info(f"sub-{str(subj).zfill(3):<7} {true_label_str:<15} {n_pd_votes:<12} {n_ctl_votes:<12} {pred_label_str:<12} {correct_str:<10}")
    
    # Subject-level accuracy (majority vote)
    subject_accuracy = np.mean(list(subject_corrects.values()))
    logger.info("-" * 80)
    logger.info(f"Subject-Level Accuracy (Majority Vote): {subject_accuracy:.4f}")
    
    # Calculate subject-level metrics
    subject_y_true = np.array([subject_to_class[subj] for subj in sorted(holdout_subjects)])
    subject_y_pred = np.array([subject_preds[int(subj)] for subj in sorted(holdout_subjects)])
    
    subject_metrics = calculate_metrics(subject_y_true, subject_y_pred)
    logger.info(f"Subject-Level Sensitivity: {subject_metrics['sensitivity']:.4f}")
    logger.info(f"Subject-Level Specificity: {subject_metrics['specificity']:.4f}")
    logger.info(f"Subject-Level F1: {subject_metrics['f1_score']:.4f}")
    
    results["subject_level"]["majority_vote"] = {
        "accuracy": subject_accuracy,
        "sensitivity": subject_metrics["sensitivity"],
        "specificity": subject_metrics["specificity"],
        "f1_score": subject_metrics["f1_score"],
        "subject_votes": subject_votes,
        "subject_predictions": subject_preds,
    }
    
    # Subject-level aggregation: Mean Probability
    logger.info(f"\nHoldout Subject-Level Evaluation (Mean Probability):")
    logger.info(f"{'Subject':<10} {'True Label':<15} {'Mean Prob PD':<15} {'Predicted':<12} {'Correct':<10}")
    logger.info("-" * 80)
    
    subject_mean_probs = {}
    subject_preds_prob = {}
    subject_corrects_prob = {}
    
    for subj in sorted(holdout_subjects):
        subj_mask = groups_holdout == subj
        subj_y_true = y_holdout[subj_mask]
        subj_y_proba = y_proba_holdout[subj_mask]
        
        # True label
        true_label = int(subj_y_true[0])
        
        # Mean probability for PD class
        mean_prob_pd = np.mean(subj_y_proba)
        
        # Predict PD if mean prob > 0.5, else CTL
        predicted = 1 if mean_prob_pd > 0.5 else 0
        
        is_correct = (predicted == true_label)
        
        # Convert subj to Python int for JSON serialization
        subj_key = int(subj)
        subject_mean_probs[subj_key] = float(mean_prob_pd)
        subject_preds_prob[subj_key] = int(predicted)
        subject_corrects_prob[subj_key] = is_correct
        
        true_label_str = "PD" if true_label == 1 else "CTL"
        pred_label_str = "PD" if predicted == 1 else "CTL"
        correct_str = "✓" if is_correct else "✗"
        
        logger.info(f"sub-{str(subj).zfill(3):<7} {true_label_str:<15} {mean_prob_pd:<15.4f} {pred_label_str:<12} {correct_str:<10}")
    
    # Subject-level accuracy (mean probability)
    subject_accuracy_prob = np.mean(list(subject_corrects_prob.values()))
    logger.info("-" * 80)
    logger.info(f"Subject-Level Accuracy (Mean Probability): {subject_accuracy_prob:.4f}")
    
    # Calculate subject-level metrics (mean probability)
    subject_y_pred_prob = np.array([subject_preds_prob[int(subj)] for subj in sorted(holdout_subjects)])
    subject_metrics_prob = calculate_metrics(subject_y_true, subject_y_pred_prob)
    logger.info(f"Subject-Level Sensitivity: {subject_metrics_prob['sensitivity']:.4f}")
    logger.info(f"Subject-Level Specificity: {subject_metrics_prob['specificity']:.4f}")
    logger.info(f"Subject-Level F1: {subject_metrics_prob['f1_score']:.4f}")
    
    results["subject_level"]["mean_probability"] = {
        "accuracy": subject_accuracy_prob,
        "sensitivity": subject_metrics_prob["sensitivity"],
        "specificity": subject_metrics_prob["specificity"],
        "f1_score": subject_metrics_prob["f1_score"],
        "subject_mean_probs": subject_mean_probs,
        "subject_predictions": subject_preds_prob,
    }
    
    return results


# ============================================================
# Main Training Function
# ============================================================

def main():
    logger.info("=" * 75)
    logger.info("KNN TRAINING — TARGET + NOVELTY CONCATENATED PREPROCESSING")
    logger.info("Version: Conservative (K=9, Manhattan, 80 features)")
    logger.info("WITH 10-SUBJECT HOLDOUT TEST SET")
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
    # SELECT HOLDOUT SUBJECTS
    # ========================================================================
    logger.info(f"\n{'='*75}")
    logger.info("HOLDOUT SUBJECT SELECTION")
    logger.info(f"{'='*75}")
    
    holdout_subject_ids, holdout_mask, subject_to_class = select_holdout_subjects(
        groups, y, n_pd=N_HOLDOUT_PD, n_ctl=N_HOLDOUT_CTL, random_state=HOLDOUT_RANDOM_STATE
    )
    
    # Create training pool (non-holdout data)
    train_mask = ~holdout_mask
    X_train_pool = X[train_mask]
    y_train_pool = y[train_mask]
    groups_train_pool = groups[train_mask]
    
    X_holdout = X[holdout_mask]
    y_holdout = y[holdout_mask]
    groups_holdout = groups[holdout_mask]
    
    logger.info(f"\nData split:")
    logger.info(f"  Training pool: {len(y_train_pool)} epochs ({np.sum(y_train_pool == 1)} PD, {np.sum(y_train_pool == 0)} CTL)")
    logger.info(f"  Holdout set: {len(y_holdout)} epochs ({np.sum(y_holdout == 1)} PD, {np.sum(y_holdout == 0)} CTL)")
    logger.info(f"  Training pool subjects: {len(np.unique(groups_train_pool))}")
    logger.info(f"  Holdout subjects: {len(np.unique(groups_holdout))}")

    # ========================================================================
    # TRAINING WITH FEATURE SELECTION & SCALING INSIDE CV FOLDS
    # ========================================================================
    logger.info(f"\n{'='*75}")
    logger.info("CROSS-VALIDATION ON TRAINING POOL")
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
    logger.info(f"  Cross-validation: {N_SPLITS}-Fold StratifiedGroupKFold on training pool")
    logger.info(f"  Data leakage prevention: Feature selection & scaling inside folds only\n")

    all_metrics = []
    all_cms = []
    all_fold_accs = []
    all_roc_data = []
    all_fold_results = {}
    fold_idx = 0

    # Cross-validation setup (on training pool only)
    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # Cross-validation loop
    for train_idx, val_idx in skf.split(X_train_pool, y_train_pool, groups_train_pool):
        fold_idx += 1
        logger.info(f"\n{'='*75}")
        logger.info(f"FOLD {fold_idx}/{N_SPLITS}")
        logger.info(f"{'='*75}")

        X_train, X_val = X_train_pool[train_idx], X_train_pool[val_idx]
        y_train, y_val = y_train_pool[train_idx], y_train_pool[val_idx]

        n_train_pd = np.sum(y_train == 1)
        n_train_ctl = np.sum(y_train == 0)
        n_val_pd = np.sum(y_val == 1)
        n_val_ctl = np.sum(y_val == 0)

        logger.info(f"\nData split:")
        logger.info(f"  Train: {len(y_train)} epochs ({n_train_pd} PD, {n_train_ctl} CTL) - {n_train_pd/len(y_train):.1%} PD")
        logger.info(f"  Val:   {len(y_val)} epochs ({n_val_pd} PD, {n_val_ctl} CTL) - {n_val_pd/len(y_val):.1%} PD")
        logger.info(f"  Train subjects: {len(np.unique(groups_train_pool[train_idx]))}")
        logger.info(f"  Val subjects: {len(np.unique(groups_train_pool[val_idx]))}")

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
            "n_train_subjects": len(np.unique(groups_train_pool[train_idx])),
            "n_val_subjects": len(np.unique(groups_train_pool[val_idx])),
            "n_features_input": X_train_pool.shape[1],
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

    # Calculate mean CV metrics
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

    # ========================================================================
    # TRAIN FINAL MODEL ON ALL TRAINING POOL DATA
    # ========================================================================
    logger.info(f"\n{'='*75}")
    logger.info("FINAL MODEL TRAINING (On All Training Pool Data)")
    logger.info(f"{'='*75}\n")
    
    logger.info(f"Selecting features on entire training pool...")
    X_train_pool_sel, _, final_sel_indices, final_mi_scores = select_features_inside_fold(
        X_train_pool, y_train_pool, X_train_pool, n_features=N_FEATURES
    )
    logger.info(f"  ✓ Selected {len(final_sel_indices)} features")
    
    logger.info(f"Scaling entire training pool...")
    X_train_pool_scaled, _, final_scaler = scale_inside_fold(X_train_pool_sel, X_train_pool_sel)
    logger.info(f"  ✓ Scaler fitted on training pool")
    
    logger.info(f"Training final KNN model...")
    final_model = KNeighborsClassifier(
        n_neighbors=KNN_K,
        metric=KNN_METRIC,
        weights=KNN_WEIGHTS,
        n_jobs=-1
    )
    final_model.fit(X_train_pool_scaled, y_train_pool)
    logger.info(f"  ✓ Final KNN model trained on {len(y_train_pool)} epochs from training pool")

    # ========================================================================
    # EVALUATE ON HOLDOUT TEST SET
    # ========================================================================
    logger.info(f"\n{'='*75}")
    logger.info("HOLDOUT TEST SET RESULTS - 10 SUBJECTS")
    logger.info(f"{'='*75}")
    
    # Apply feature selection and scaling to holdout set
    X_holdout_sel = X_holdout[:, final_sel_indices]
    X_holdout_scaled = final_scaler.transform(X_holdout_sel)
    
    # Make predictions on holdout set
    y_holdout_pred = final_model.predict(X_holdout_scaled)
    y_holdout_proba = final_model.predict_proba(X_holdout_scaled)[:, 1]
    
    # Evaluate holdout performance
    holdout_results = evaluate_holdout_subjects(
        X_holdout, y_holdout, groups_holdout, y_holdout_pred, y_holdout_proba, subject_to_class
    )
    
    all_fold_results["holdout_results"] = holdout_results

    # Generate final report
    logger.info(f"\nGenerating final report...")

    # Get holdout subjects for report generation
    holdout_subjects = np.unique(groups_holdout)

    report_path = OUT_DIR / "FINAL_REPORT_KNN_TARGET_NOVELTY_K9_MANHATTAN_80F_HOLDOUT10.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("KNN CLASSIFICATION MODEL - FINAL REPORT\n")
        f.write("Target + Novelty Concatenated Preprocessing\n")
        f.write("Version: Conservative (K=9, Manhattan, 80 features)\n")
        f.write("WITH 10-SUBJECT HOLDOUT TEST SET\n")
        f.write("=" * 80 + "\n\n")

        f.write("DATASET SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Feature mode: {feature_mode}\n")
        f.write(f"Total epochs (full dataset): {X.shape[0]}\n")
        f.write(f"Features (input): {X.shape[1]}\n")
        f.write(f"  → Target features: {X.shape[1]//2}\n")
        f.write(f"  → Novelty features: {X.shape[1]//2}\n")
        f.write(f"Total subjects: {len(np.unique(groups))}\n")
        f.write(f"Full dataset class balance:\n")
        f.write(f"  CTL: {np.sum(y == 0)} epochs\n")
        f.write(f"  PD: {np.sum(y == 1)} epochs\n")
        f.write(f"  PD percentage: {np.sum(y == 1) / len(y):.2%}\n\n")

        f.write("HOLDOUT CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Holdout subjects (random_state={HOLDOUT_RANDOM_STATE}): {N_HOLDOUT_SUBJECTS} subjects\n")
        f.write(f"  → {N_HOLDOUT_PD} PD subjects: {sorted([s for s in holdout_subject_ids if subject_to_class[s] == 1])}\n")
        f.write(f"  → {N_HOLDOUT_CTL} CTL subjects: {sorted([s for s in holdout_subject_ids if subject_to_class[s] == 0])}\n\n")

        f.write("DATA SPLIT\n")
        f.write("-" * 80 + "\n")
        f.write(f"Training pool (used for CV and final model training):\n")
        f.write(f"  Epochs: {len(y_train_pool)}\n")
        f.write(f"  PD: {np.sum(y_train_pool == 1)}, CTL: {np.sum(y_train_pool == 0)}\n")
        f.write(f"  Subjects: {len(np.unique(groups_train_pool))}\n")
        f.write(f"  PD balance: {np.sum(y_train_pool == 1)/len(y_train_pool):.2%}\n\n")
        f.write(f"Holdout test set (held out until final evaluation):\n")
        f.write(f"  Epochs: {len(y_holdout)}\n")
        f.write(f"  PD: {np.sum(y_holdout == 1)}, CTL: {np.sum(y_holdout == 0)}\n")
        f.write(f"  Subjects: {len(np.unique(groups_holdout))}\n")
        f.write(f"  PD balance: {np.sum(y_holdout == 1)/len(y_holdout):.2%}\n\n")

        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Algorithm: K-Nearest Neighbors\n")
        f.write(f"K value: {KNN_K}\n")
        f.write(f"Distance metric: {KNN_METRIC}\n")
        f.write(f"Weights: {KNN_WEIGHTS}\n")
        f.write(f"Cross-validation: {N_SPLITS}-Fold StratifiedGroupKFold (on training pool)\n\n")

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

        f.write("PER-FOLD CROSS-VALIDATION PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Fold':<6} {'Train Acc':<12} {'Val Acc':<12} {'Val Sens':<12} {'Val Spec':<12} {'Val AUC':<12}\n")
        f.write("-" * 80 + "\n")
        for idx, row in metrics_df.iterrows():
            val_sens = all_fold_results[f"fold_{int(row['fold'])}"]["val_metrics"]["sensitivity"]
            val_spec = all_fold_results[f"fold_{int(row['fold'])}"]["val_metrics"]["specificity"]
            f.write(f"{int(row['fold']):<6} {row['train_accuracy']:<12.4f} {row['val_accuracy']:<12.4f} {val_sens:<12.4f} {val_spec:<12.4f} {row['val_roc_auc']:<12.4f}\n")
        f.write("\n")

        f.write("MEAN CROSS-VALIDATION PERFORMANCE (5-FOLD, TRAINING POOL)\n")
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
        f.write("HOLDOUT TEST SET RESULTS - 10 SUBJECTS\n")
        f.write("=" * 80 + "\n\n")

        # Holdout epoch-level results
        f.write("HOLDOUT EPOCH-LEVEL EVALUATION\n")
        f.write("-" * 80 + "\n")
        holdout_epoch_metrics = holdout_results["epoch_level"]
        f.write(f"Accuracy:           {holdout_epoch_metrics['accuracy']:.4f}\n")
        f.write(f"Sensitivity (Recall): {holdout_epoch_metrics['sensitivity']:.4f}\n")
        f.write(f"Specificity:        {holdout_epoch_metrics['specificity']:.4f}\n")
        f.write(f"Precision:          {holdout_epoch_metrics['precision']:.4f}\n")
        f.write(f"F1 Score:           {holdout_epoch_metrics['f1_score']:.4f}\n")
        f.write(f"ROC-AUC:            {holdout_epoch_metrics.get('roc_auc', 0):.4f}\n\n")

        # Holdout confusion matrix
        if "confusion_matrix" in holdout_epoch_metrics:
            cm = np.array(holdout_epoch_metrics["confusion_matrix"])
            tn, fp, fn, tp = cm.ravel()
            f.write("Confusion Matrix (Holdout Epochs):\n")
            f.write(f"  TN={int(tn):<3}  FP={int(fp):<3}  (Predicted CTL)\n")
            f.write(f"  FN={int(fn):<3}  TP={int(tp):<3}  (Predicted PD)\n")
            f.write("  (True CTL)  (True PD)\n\n")

        # Holdout subject-level results (majority vote)
        f.write("HOLDOUT SUBJECT-LEVEL EVALUATION (Majority Vote)\n")
        f.write("-" * 80 + "\n")
        mv_results = holdout_results["subject_level"]["majority_vote"]
        
        # Subject-level majority vote table
        f.write(f"{'Subject':<12} {'True Label':<15} {'Predicted':<15} {'Epochs':<10} {'PD Votes':<12} {'CTL Votes':<15}\n")
        f.write("-" * 80 + "\n")
        for subj in sorted(holdout_subjects):
            subj_key = int(subj)
            true_label = subject_to_class[subj_key]
            true_label_str = "PD" if true_label == 1 else "CTL"
            pred_label = mv_results["subject_predictions"][subj_key]
            pred_label_str = "PD" if pred_label == 1 else "CTL"
            pd_votes = mv_results["subject_votes"][subj_key]["pd_votes"]
            ctl_votes = mv_results["subject_votes"][subj_key]["ctl_votes"]
            n_epochs = np.sum(groups_holdout == subj)
            f.write(f"sub-{str(subj_key).zfill(3):<9} {true_label_str:<15} {pred_label_str:<15} {n_epochs:<10} {pd_votes:<12} {ctl_votes:<15}\n")
        f.write("\n")
        
        f.write("Subject-level Majority Vote Performance:\n")
        f.write(f"  Accuracy:    {mv_results['accuracy']:.4f}\n")
        f.write(f"  Sensitivity: {mv_results['sensitivity']:.4f}\n")
        f.write(f"  Specificity: {mv_results['specificity']:.4f}\n")
        f.write(f"  F1 Score:    {mv_results['f1_score']:.4f}\n")
        
        # Calculate subject-level confusion matrix for majority vote
        subject_y_true = np.array([subject_to_class[int(subj)] for subj in sorted(holdout_subjects)])
        subject_y_pred_mv = np.array([mv_results["subject_predictions"][int(subj)] for subj in sorted(holdout_subjects)])
        subject_cm_mv = confusion_matrix(subject_y_true, subject_y_pred_mv, labels=[0, 1])
        tn, fp, fn, tp = subject_cm_mv.ravel()
        f.write(f"  Confusion Matrix: TN={int(tn)}, FP={int(fp)}, FN={int(fn)}, TP={int(tp)}\n\n")

        # Holdout subject-level results (mean probability)
        f.write("HOLDOUT SUBJECT-LEVEL EVALUATION (Mean Probability, threshold=0.5)\n")
        f.write("-" * 80 + "\n")
        mp_results = holdout_results["subject_level"]["mean_probability"]
        
        # Subject-level mean probability table
        f.write(f"{'Subject':<12} {'True Label':<15} {'Predicted':<15} {'Mean Prob':<15} {'Epochs':<10}\n")
        f.write("-" * 80 + "\n")
        for subj in sorted(holdout_subjects):
            subj_key = int(subj)
            true_label = subject_to_class[subj_key]
            true_label_str = "PD" if true_label == 1 else "CTL"
            pred_label = mp_results["subject_predictions"][subj_key]
            pred_label_str = "PD" if pred_label == 1 else "CTL"
            mean_prob = mp_results["subject_mean_probs"][subj_key]
            n_epochs = np.sum(groups_holdout == subj)
            f.write(f"sub-{str(subj_key).zfill(3):<9} {true_label_str:<15} {pred_label_str:<15} {mean_prob:<15.4f} {n_epochs:<10}\n")
        f.write("\n")
        
        f.write("Subject-level Mean Probability Performance:\n")
        f.write(f"  Accuracy:    {mp_results['accuracy']:.4f}\n")
        f.write(f"  Sensitivity: {mp_results['sensitivity']:.4f}\n")
        f.write(f"  Specificity: {mp_results['specificity']:.4f}\n")
        f.write(f"  F1 Score:    {mp_results['f1_score']:.4f}\n")
        
        # Calculate subject-level confusion matrix for mean probability
        subject_y_pred_mp = np.array([mp_results["subject_predictions"][int(subj)] for subj in sorted(holdout_subjects)])
        subject_cm_mp = confusion_matrix(subject_y_true, subject_y_pred_mp, labels=[0, 1])
        tn, fp, fn, tp = subject_cm_mp.ravel()
        f.write(f"  Confusion Matrix: TN={int(tn)}, FP={int(fp)}, FN={int(fn)}, TP={int(tp)}\n\n")

        f.write("=" * 80 + "\n")
        f.write("SUMMARY & INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Cross-Validation Performance (on training pool - non-holdout subjects only):\n")
        f.write(f"The conservative KNN model (K=9, Manhattan, 80 features) achieved a mean\n")
        f.write(f"accuracy of {mean_val_acc:.2%} across 5-fold cross-validation, with ROC-AUC\n")
        f.write(f"of {mean_val_auc:.3f}, using target + novelty concatenated preprocessing.\n\n")
        f.write(f"Key CV findings:\n")
        f.write(f"• Sensitivity (recall): {mean_val_rec:.2%} - captures {mean_val_rec:.0%} of PD cases\n")
        f.write(f"• Specificity: {mean_val_spec:.2%} - correctly identifies {mean_val_spec:.0%} of controls\n")
        f.write(f"• False Negative Rate: {mean_val_fnr:.2%} - misses {mean_val_fnr:.0%} of actual PD cases\n")
        f.write(f"• F1 Score: {mean_val_f1:.4f} - balanced precision-recall performance\n\n")
        
        f.write("Holdout Test Performance (final external validation on 10 held-out subjects):\n")
        f.write(f"On the {N_HOLDOUT_SUBJECTS} held-out subjects (never seen during model development):\n\n")
        f.write(f"Epoch-level results:\n")
        f.write(f"• Accuracy: {holdout_epoch_metrics['accuracy']:.2%}\n")
        f.write(f"• Sensitivity: {holdout_epoch_metrics['sensitivity']:.2%}\n")
        f.write(f"• Specificity: {holdout_epoch_metrics['specificity']:.2%}\n\n")
        f.write(f"Subject-level results (Majority Vote):\n")
        f.write(f"• Accuracy: {mv_results['accuracy']:.2%}\n")
        f.write(f"• Sensitivity: {mv_results['sensitivity']:.2%}\n")
        f.write(f"• Specificity: {mv_results['specificity']:.2%}\n\n")
        
        f.write(f"Data leakage prevention:\n")
        f.write(f"• Holdout subjects ({N_HOLDOUT_SUBJECTS} subjects) are NEVER used in cross-validation\n")
        f.write(f"• Holdout subjects are NEVER used when fitting feature selectors or scalers\n")
        f.write(f"• Feature selection (mutual information) is performed independently within\n")
        f.write(f"  each CV fold using training data only\n")
        f.write(f"• Final model feature selection and scaling is fit ONLY on non-holdout data\n")
        f.write(f"• This ensures unbiased performance estimates on the holdout set\n")
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
        "version": "conservative_k9_manhattan_80f_holdout10",
        "holdout_n_subjects": N_HOLDOUT_SUBJECTS,
        "holdout_n_pd": N_HOLDOUT_PD,
        "holdout_n_ctl": N_HOLDOUT_CTL,
        "holdout_random_state": HOLDOUT_RANDOM_STATE,
        "holdout_subject_ids": [int(s) for s in holdout_subject_ids],
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
