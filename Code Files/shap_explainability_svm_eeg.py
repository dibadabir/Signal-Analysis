# %% [markdown]
# # SHAP Explainability Report — EEG-based SVM for Parkinson's Disease Detection
#
# Run this file in VS Code with the **Python** extension installed.
# Each `# %%` block is an interactive cell — click "Run Cell" above each block,
# or press **Shift+Enter** to run the current cell and advance.
#
# **What this script produces:**
# - Global feature importance (beeswarm and bar plots)
# - Per-class SHAP analysis (PD OFF vs Control)
# - Waterfall plots for individual epoch predictions
# - Dependence plots for top EEG features
# - Subject-level SHAP aggregation + heatmap
# - EEG frequency-band and channel contribution summaries
# - Interactive HTML force plot + CSV exports
#
# **Prerequisites:**
#   pip install shap mne scikit-learn pandas numpy matplotlib joblib openpyxl

# %%
# ============================================================
# 1. Imports
# ============================================================
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import shap
import mne

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score

print("Imports OK")

# %%
# ============================================================
# 2. Configuration — update paths to match your environment
# ============================================================

# --- Paths ---
OUT_DIR      = Path(r"/Users/dibadabiransari/Desktop/EEG Project/Correct")
CACHE_EPOCHS = OUT_DIR / "cache_epochs_binary"
SVM_RESULTS  = OUT_DIR / "svm_final_clean_attempt"
SHAP_DIR     = OUT_DIR / "shap_explainability"
SHAP_DIR.mkdir(parents=True, exist_ok=True)

# Saved model from your SVM notebook
MODEL_PATH = SVM_RESULTS / "best_svm_model.joblib"

# --- Feature / model settings (must match your SVM notebook) ---
BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8,  12),
    "beta":  (12, 30),
}
SELECTED_CHANNELS = ["F3", "F4", "Fz", "C3", "C4", "Cz", "P3", "P4", "Pz"]
ERP_WINDOWS = [
    ("erp_250_400",  0.25, 0.40),
    ("erp_400_700",  0.40, 0.70),
    ("erp_700_1000", 0.70, 1.00),
]
USE_PD_LABEL  = "OFF"
USE_CTL_LABEL = "CTL"
K_BEST_FEATURES = 60
RANDOM_STATE    = 42
TEST_SIZE       = 0.20

# --- SHAP sampling (KernelExplainer is slow — tune to your hardware) ---
N_BACKGROUND   = 100   # background samples for KernelExplainer
N_SHAP_SAMPLES = 200   # test epochs to explain (lower = faster)

print(f"SHAP outputs → {SHAP_DIR}")

# %%
# ============================================================
# 3. Feature-extraction helpers (identical to your SVM notebook)
# ============================================================

def parse_file_label(file_path: Path):
    stem  = file_path.name.replace("-epo.fif", "")
    parts = stem.split("_")
    if len(parts) != 2:
        return None, None
    return parts[0].replace("sub-", ""), parts[1]


def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=(b != 0))


def compute_bandpower_maps(data, sfreq, bands):
    """data: (n_epochs, n_channels, n_times) → dict band → (n_epochs, n_channels)"""
    n_times   = data.shape[-1]
    n_per_seg = min(int(sfreq), n_times)
    n_fft     = min(512, max(128, n_times))
    n_overlap = n_per_seg // 2 if n_per_seg >= 4 else 0
    band_dict = {}
    for band_name, (fmin, fmax) in bands.items():
        psd_out = mne.time_frequency.psd_array_welch(
            data, sfreq=sfreq, fmin=fmin, fmax=fmax,
            n_per_seg=n_per_seg, n_fft=n_fft, n_overlap=n_overlap,
            average="mean", verbose=False)
        psd = psd_out[0] if isinstance(psd_out, tuple) else psd_out
        band_dict[band_name] = psd.mean(axis=-1)
    return band_dict


def extract_features(epochs, bands, selected_channels, erp_windows):
    epochs   = epochs.copy().pick(selected_channels)
    data     = epochs.get_data().astype(np.float32)
    sfreq    = float(epochs.info["sfreq"])
    ch_names = epochs.ch_names
    times    = epochs.times

    blocks, names = [], []
    band_dict = compute_bandpower_maps(data, sfreq, bands)

    # 1 Absolute bandpower
    for bn in bands:
        blocks.append(band_dict[bn])
        names.extend([f"{ch}_{bn}_abs" for ch in ch_names])

    # 2 Relative bandpower
    total = sum(band_dict[bn] for bn in bands)
    for bn in bands:
        rel = safe_divide(band_dict[bn], total)
        blocks.append(rel)
        names.extend([f"{ch}_{bn}_rel" for ch in ch_names])

    # 3 Ratios
    for rname, (num_b, den_b) in [
        ("theta_beta_ratio",  ("theta", "beta")),
        ("alpha_theta_ratio", ("alpha", "theta")),
        ("alpha_beta_ratio",  ("alpha", "beta")),
    ]:
        ratio = safe_divide(band_dict[num_b], band_dict[den_b])
        blocks.append(ratio)
        names.extend([f"{ch}_{rname}" for ch in ch_names])

    # 4 ERP window means
    for wname, t1, t2 in erp_windows:
        idx = np.where((times >= t1) & (times <= t2))[0]
        if len(idx):
            wm = data[:, :, idx].mean(axis=2)
            blocks.append(wm)
            names.extend([f"{ch}_{wname}_mean" for ch in ch_names])

    # 5 Global summaries
    for gname, arr in [
        ("global_delta_rel", safe_divide(band_dict["delta"], total).mean(axis=1, keepdims=True)),
        ("global_theta_rel", safe_divide(band_dict["theta"], total).mean(axis=1, keepdims=True)),
        ("global_alpha_rel", safe_divide(band_dict["alpha"], total).mean(axis=1, keepdims=True)),
        ("global_beta_rel",  safe_divide(band_dict["beta"],  total).mean(axis=1, keepdims=True)),
    ]:
        blocks.append(arr)
        names.append(gname)

    X = np.concatenate(blocks, axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


def load_dataset(cache_dir: Path, pd_label="OFF", ctl_label="CTL"):
    X_list, y_list, groups = [], [], []
    feature_names = None
    meta_rows     = []

    for fp in sorted(cache_dir.glob("sub-*-epo.fif")):
        subj, label = parse_file_label(fp)
        if subj is None or label not in {pd_label, ctl_label}:
            continue
        y = 1 if label == pd_label else 0
        try:
            epo   = mne.read_epochs(fp, preload=True, verbose=False)
            avail = [c for c in SELECTED_CHANNELS if c in epo.ch_names]
            if len(avail) < 6:
                continue
            Xs, fn = extract_features(epo, BANDS, avail, ERP_WINDOWS)
            if feature_names is None:
                feature_names = fn
            X_list.append(Xs)
            y_list.append(np.full(len(Xs), y, dtype=int))
            groups.extend([subj] * len(Xs))
            meta_rows.append(dict(subject=subj, label=label, cls=y, n_epochs=len(epo)))
        except Exception as e:
            print(f"  Skipping {fp.name}: {e}")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y, np.array(groups), feature_names, pd.DataFrame(meta_rows)


print("Helpers defined.")

# %%
# ============================================================
# 4. Load dataset and rebuild the same train / test split
# ============================================================

print("Loading dataset from cached epochs …")
X, y, groups, feature_names, meta_df = load_dataset(CACHE_EPOCHS)
feature_names = np.array(feature_names)

print(f"  Total epochs  : {len(X)}")
print(f"  Total features: {X.shape[1]}")
print(f"  Class counts  : CTL={np.sum(y==0)}  PD_OFF={np.sum(y==1)}")
print(f"  Subjects      : {len(np.unique(groups))}")

# Rebuild the identical test split used in the SVM notebook
subject_df = (
    pd.DataFrame({"subject": groups, "y": y})
    .groupby("subject")["y"].first()
    .reset_index()
)
subj_ids = subject_df["subject"].values
subj_y   = subject_df["y"].values

gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
tv_idx, te_idx = next(gss.split(subj_ids, subj_y, groups=subj_ids))
test_subjects  = subj_ids[te_idx]
train_subjects = subj_ids[tv_idx]

train_mask = np.isin(groups, train_subjects)
test_mask  = np.isin(groups, test_subjects)

X_train, y_train, g_train = X[train_mask], y[train_mask], groups[train_mask]
X_test,  y_test,  g_test  = X[test_mask],  y[test_mask],  groups[test_mask]

print(f"\n  Train: {len(X_train)} epochs  |  Test: {len(X_test)} epochs")
print(f"  Test subjects: {np.unique(g_test).tolist()}")

pd.DataFrame({"feature": feature_names}).to_csv(SHAP_DIR / "feature_names.csv", index=False)

# %%
# ============================================================
# 5. Load (or retrain) the best SVM model
# ============================================================

if MODEL_PATH.exists():
    best_model = joblib.load(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")
else:
    print("Model file not found — retraining with best known params …")
    k_best = min(K_BEST_FEATURES, X_train.shape[1])
    best_model = Pipeline([
        ("var_thresh", VarianceThreshold()),
        ("scaler",     StandardScaler()),
        ("select",     SelectKBest(score_func=f_classif, k=k_best)),
        ("pca",        PCA(n_components=0.95, svd_solver="full")),
        ("svm",        SVC(kernel="rbf", C=0.5, gamma="scale",
                           class_weight="balanced", probability=True,
                           random_state=RANDOM_STATE)),
    ])
    best_model.fit(X_train, y_train)
    print("Retrain complete.")

# Sanity check
test_proba = best_model.predict_proba(X_test)[:, 1]
test_pred  = (test_proba >= 0.44).astype(int)
print(f"\nTest ROC-AUC (sanity): {roc_auc_score(y_test, test_proba):.4f}")
print(f"Test Accuracy (0.44) : {accuracy_score(y_test, test_pred):.4f}")

# %%
# ============================================================
# 6. Set up SHAP KernelExplainer
#
# We treat the FULL pipeline as a black box so that SHAP values
# are returned in the original 130-feature space — interpretable
# as named EEG features.
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)

# Background: sample from training data, then summarise with k-means
bg_idx       = rng.choice(len(X_train), size=min(N_BACKGROUND, len(X_train)), replace=False)
X_background = X_train[bg_idx]
X_bg_summary = shap.kmeans(X_background, 50)
print(f"Background: {len(X_background)} samples → 50 k-means centroids")

# Prediction function exposed to SHAP: returns P(PD OFF)
def predict_pd_prob(X_arr):
    return best_model.predict_proba(X_arr)[:, 1]

print("Building KernelExplainer …")
explainer = shap.KernelExplainer(predict_pd_prob, X_bg_summary)
print("KernelExplainer ready.")

# Sample test epochs for explanation
n_explain  = min(N_SHAP_SAMPLES, len(X_test))
exp_idx    = rng.choice(len(X_test), size=n_explain, replace=False)
X_explain  = X_test[exp_idx]
y_explain  = y_test[exp_idx]
g_explain  = g_test[exp_idx]

print(f"\nWill explain {n_explain} test epochs  "
      f"(CTL={np.sum(y_explain==0)}  PD_OFF={np.sum(y_explain==1)})")

# %%
# ============================================================
# 7. Compute SHAP values
#    (~5–30 min depending on N_SHAP_SAMPLES and hardware)
# ============================================================

print("Computing SHAP values — please wait …")
shap_values    = np.array(explainer.shap_values(X_explain, nsamples="auto"))
expected_value = float(explainer.expected_value)

print(f"\nSHAP values shape : {shap_values.shape}")
print(f"E[f(x)] (baseline): {expected_value:.4f}")

# Save raw SHAP values as CSV
shap_df = pd.DataFrame(shap_values, columns=feature_names)
shap_df["y_true"]  = y_explain
shap_df["y_prob"]  = best_model.predict_proba(X_explain)[:, 1]
shap_df["subject"] = g_explain
shap_df.to_csv(SHAP_DIR / "shap_values_per_epoch.csv", index=False)
print("Raw SHAP values saved.")

# Pre-compute reused quantities
mean_abs_shap = np.abs(shap_values).mean(axis=0)
proba_explain = shap_df["y_prob"].values

importance_df = (
    pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
importance_df.to_csv(SHAP_DIR / "feature_importance_shap.csv", index=False)

# %%
# ============================================================
# 8. Global Summary — Beeswarm Plot
#
# Each dot = one epoch.
# x-axis = SHAP value (positive → pushes toward PD OFF).
# Colour  = feature value (red = high, blue = low).
# Sorted by mean |SHAP|.
# ============================================================

plt.figure(figsize=(12, 9))
shap.summary_plot(
    shap_values,
    X_explain,
    feature_names=feature_names.tolist(),
    max_display=20,
    show=False,
)
plt.title(
    "SHAP Global Feature Importance\n(positive SHAP → pushes toward PD OFF prediction)",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(SHAP_DIR / "shap_beeswarm_top20.png", dpi=150, bbox_inches="tight")
plt.show()
print("Beeswarm plot saved.")

# %%
# ============================================================
# 9. Feature Importance — Bar Plot (mean |SHAP|)
# ============================================================

def feature_colour(fname):
    if "_abs"   in fname: return "#2196F3"
    if "_rel"   in fname: return "#4CAF50"
    if "ratio"  in fname: return "#FF9800"
    if "erp_"   in fname: return "#9C27B0"
    if "global_" in fname: return "#F44336"
    return "#607D8B"

top25   = importance_df.head(25)
colours = [feature_colour(f) for f in top25["feature"]]

legend_handles = [
    mpatches.Patch(color="#2196F3", label="Absolute bandpower"),
    mpatches.Patch(color="#4CAF50", label="Relative bandpower"),
    mpatches.Patch(color="#FF9800", label="Band ratio"),
    mpatches.Patch(color="#9C27B0", label="ERP window mean"),
    mpatches.Patch(color="#F44336", label="Global summary"),
]

fig, ax = plt.subplots(figsize=(11, 8))
ax.barh(top25["feature"][::-1], top25["mean_abs_shap"][::-1],
        color=colours[::-1], edgecolor="white", linewidth=0.5)
ax.set_xlabel("Mean |SHAP value|  (impact on model output)", fontsize=12)
ax.set_title("Top 25 EEG Features by SHAP Importance\n(SVM PD-OFF vs Control)", fontsize=13)
ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(SHAP_DIR / "shap_bar_top25.png", dpi=150, bbox_inches="tight")
plt.show()
print("Bar importance plot saved.")
print(importance_df.head(20).to_string(index=False))

# %%
# ============================================================
# 10. Per-class SHAP Analysis — CTL vs PD OFF
# ============================================================

ctl_mask = (y_explain == 0)
pd_mask  = (y_explain == 1)

mean_shap_ctl = np.abs(shap_values[ctl_mask]).mean(axis=0)
mean_shap_pd  = np.abs(shap_values[pd_mask]).mean(axis=0)

class_importance = pd.DataFrame({
    "feature":            feature_names,
    "mean_abs_shap_CTL":  mean_shap_ctl,
    "mean_abs_shap_PD":   mean_shap_pd,
    "pd_over_ctl_ratio":  np.where(
        mean_shap_ctl > 0,
        mean_shap_pd / (mean_shap_ctl + 1e-10),
        np.nan,
    ),
}).sort_values("mean_abs_shap_PD", ascending=False)
class_importance.to_csv(SHAP_DIR / "shap_per_class_importance.csv", index=False)

top_n    = 15
top_feat = class_importance.head(top_n)
x = np.arange(top_n)
w = 0.38

fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(x - w/2, top_feat["mean_abs_shap_CTL"], width=w,
       label="Control", color="#1976D2", alpha=0.85)
ax.bar(x + w/2, top_feat["mean_abs_shap_PD"],  width=w,
       label="PD OFF",  color="#D32F2F", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(top_feat["feature"], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Mean |SHAP value|", fontsize=11)
ax.set_title(f"Top {top_n} Features: Control vs PD OFF SHAP Importance", fontsize=13)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(SHAP_DIR / "shap_perclass_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Per-class SHAP plot saved.")
print(class_importance.head(15).to_string(index=False))

# %%
# ============================================================
# 11. Frequency-Band Contribution Summary
# ============================================================

band_shap = {}
for band in ["delta", "theta", "alpha", "beta"]:
    mask = np.array([f"_{band}_" in n for n in feature_names])
    if mask.any():
        band_shap[band] = float(np.abs(shap_values[:, mask]).mean())

for ratio in ["theta_beta_ratio", "alpha_theta_ratio", "alpha_beta_ratio"]:
    mask = np.array([ratio in n for n in feature_names])
    if mask.any():
        band_shap[ratio] = float(np.abs(shap_values[:, mask]).mean())

for win in ["erp_250_400", "erp_400_700", "erp_700_1000"]:
    mask = np.array([win in n for n in feature_names])
    if mask.any():
        band_shap[win] = float(np.abs(shap_values[:, mask]).mean())

g_mask = np.array(["global_" in n for n in feature_names])
if g_mask.any():
    band_shap["global_summary"] = float(np.abs(shap_values[:, g_mask]).mean())

band_df = (
    pd.DataFrame(list(band_shap.items()), columns=["feature_group", "mean_abs_shap"])
    .sort_values("mean_abs_shap", ascending=False)
)
band_df.to_csv(SHAP_DIR / "shap_band_contribution.csv", index=False)

palette = [
    "#E53935","#1E88E5","#43A047","#FB8C00","#8E24AA",
    "#00ACC1","#F06292","#6D4C41","#26A69A","#78909C",
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].pie(
    band_df["mean_abs_shap"],
    labels=band_df["feature_group"],
    autopct="%1.1f%%",
    colors=palette[:len(band_df)],
    startangle=140,
    pctdistance=0.82,
)
axes[0].set_title("Proportion of SHAP Importance\nby Feature Group", fontsize=12)

axes[1].barh(
    band_df["feature_group"][::-1],
    band_df["mean_abs_shap"][::-1],
    color=palette[:len(band_df)],
)
axes[1].set_xlabel("Mean |SHAP value|", fontsize=11)
axes[1].set_title("Feature-Group SHAP Contribution", fontsize=12)
axes[1].grid(axis="x", alpha=0.3)

plt.suptitle("EEG Feature Group Contributions (SHAP)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(SHAP_DIR / "shap_band_contribution.png", dpi=150, bbox_inches="tight")
plt.show()
print(band_df.to_string(index=False))

# %%
# ============================================================
# 12. Dependence Plots — Top 5 Features
#
# Shows how each feature's raw value relates to its SHAP value.
# Colour = true class (red = PD OFF, blue = CTL).
# ============================================================

top5 = importance_df["feature"].head(5).tolist()

fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for ax, feat in zip(axes, top5):
    fi = list(feature_names).index(feat)
    fv = X_explain[:, fi]
    fs = shap_values[:, fi]

    sc = ax.scatter(fv, fs, c=y_explain, cmap="coolwarm",
                    alpha=0.6, s=18, edgecolors="none")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel(feat, fontsize=9)
    ax.set_ylabel("SHAP value", fontsize=9)
    ax.set_title(feat.replace("_", "\n"), fontsize=9)
    ax.grid(alpha=0.2)

    z = np.polyfit(fv, fs, 1)
    xseq = np.linspace(fv.min(), fv.max(), 100)
    ax.plot(xseq, np.poly1d(z)(xseq), "k-", linewidth=1.2, alpha=0.7)

cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6)
cbar.set_label("Class (0=CTL, 1=PD OFF)", fontsize=10)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["CTL", "PD OFF"])

plt.suptitle("SHAP Dependence Plots — Top 5 Features", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(SHAP_DIR / "shap_dependence_top5.png", dpi=150, bbox_inches="tight")
plt.show()
print("Dependence plots saved.")

# %%
# ============================================================
# 13. Waterfall Plots — Individual Epoch Explanations
#
# Most-confident CTL epoch and most-confident PD OFF epoch.
# ============================================================

ctl_cands  = np.where(y_explain == 0)[0]
ctl_best   = ctl_cands[np.argmin(proba_explain[ctl_cands])]

pd_cands   = np.where(y_explain == 1)[0]
pd_best    = pd_cands[np.argmax(proba_explain[pd_cands])]


def plot_waterfall(idx, label_str, save_name):
    exp_obj = shap.Explanation(
        values=shap_values[idx],
        base_values=expected_value,
        data=X_explain[idx],
        feature_names=feature_names.tolist(),
    )
    fig = plt.figure(figsize=(10, 7))
    shap.plots.waterfall(exp_obj, max_display=15, show=False)
    plt.title(
        f"SHAP Waterfall — {label_str}\n"
        f"Subject: {g_explain[idx]}  |  P(PD OFF) = {proba_explain[idx]:.3f}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(SHAP_DIR / save_name, dpi=150, bbox_inches="tight")
    plt.show()


print("=== Most confident CONTROL epoch ===")
plot_waterfall(ctl_best, "Control (CTL)", "shap_waterfall_ctl.png")

print("=== Most confident PD OFF epoch ===")
plot_waterfall(pd_best, "Parkinson's Disease OFF", "shap_waterfall_pd.png")

print("Waterfall plots saved.")

# %%
# ============================================================
# 14. Subject-level SHAP Aggregation
#
# Mean SHAP value per subject (averaged across all their epochs).
# Saved as CSV + heatmap.
# ============================================================

subj_rows = []
for subj in np.unique(g_explain):
    smask = (g_explain == subj)
    row = {
        "subject":   subj,
        "y_true":    int(y_explain[smask][0]),
        "n_epochs":  int(smask.sum()),
        "mean_prob": float(proba_explain[smask].mean()),
    }
    mean_sv = shap_values[smask].mean(axis=0)
    for fi, fn in enumerate(feature_names):
        row[fn] = float(mean_sv[fi])
    subj_rows.append(row)

subj_shap_df = pd.DataFrame(subj_rows)
subj_shap_df.to_csv(SHAP_DIR / "shap_subject_mean.csv", index=False)

print(f"Subject-level SHAP saved: {len(subj_shap_df)} subjects\n")
print(
    subj_shap_df[["subject", "y_true", "n_epochs", "mean_prob"]]
    .sort_values("mean_prob", ascending=False)
    .to_string(index=False)
)

# Heatmap: subjects × top-15 features
top15_feat  = importance_df["feature"].head(15).tolist()
hmap_data   = subj_shap_df[top15_feat].values
subj_labels = [
    f"{r['subject']} ({'PD' if r['y_true']==1 else 'CTL'})"
    for _, r in subj_shap_df.iterrows()
]

fig, ax = plt.subplots(figsize=(14, max(5, len(subj_shap_df) * 0.55)))
vmax = np.abs(hmap_data).max()
im   = ax.imshow(hmap_data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(15))
ax.set_xticklabels(top15_feat, rotation=55, ha="right", fontsize=8)
ax.set_yticks(range(len(subj_labels)))
ax.set_yticklabels(subj_labels, fontsize=9)
plt.colorbar(im, ax=ax, label="Mean SHAP value  (+ → PD OFF)", shrink=0.6)
ax.set_title("Subject-level Mean SHAP Values — Top 15 Features", fontsize=13)
plt.tight_layout()
plt.savefig(SHAP_DIR / "shap_subject_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Subject heatmap saved.")

# %%
# ============================================================
# 15. Channel-level SHAP Importance + Topographic Map
# ============================================================

ch_shap = {}
for ch in SELECTED_CHANNELS:
    mask = np.array([n.startswith(ch + "_") for n in feature_names])
    if mask.any():
        ch_shap[ch] = float(np.abs(shap_values[:, mask]).mean())

ch_df = (
    pd.DataFrame(list(ch_shap.items()), columns=["channel", "mean_abs_shap"])
    .sort_values("mean_abs_shap", ascending=False)
)
ch_df.to_csv(SHAP_DIR / "shap_channel_importance.csv", index=False)

# Approximate 2-D head positions
topo_pos = {
    "Fp1": (-0.30,  0.85), "Fp2": ( 0.30,  0.85),
    "F3":  (-0.45,  0.55), "F4":  ( 0.45,  0.55), "Fz":  ( 0.00,  0.60),
    "FC1": (-0.22,  0.35), "FC2": ( 0.22,  0.35), "FCz": ( 0.00,  0.38),
    "C3":  (-0.60,  0.00), "C4":  ( 0.60,  0.00), "Cz":  ( 0.00,  0.00),
    "P3":  (-0.45, -0.55), "P4":  ( 0.45, -0.55), "Pz":  ( 0.00, -0.55),
    "T7":  (-0.85,  0.00), "T8":  ( 0.85,  0.00),
}
max_val = ch_df["mean_abs_shap"].max()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar
colours_ch = plt.cm.Reds(ch_df["mean_abs_shap"] / max_val)
axes[0].barh(ch_df["channel"][::-1], ch_df["mean_abs_shap"][::-1],
             color=colours_ch[::-1])
axes[0].set_xlabel("Mean |SHAP value|", fontsize=11)
axes[0].set_title("Channel-level SHAP Importance", fontsize=12)
axes[0].grid(axis="x", alpha=0.3)

# Topographic bubble map
ax2 = axes[1]
ax2.add_patch(plt.Circle((0, 0), 1.0, color="lightgray", fill=False, linewidth=2))
ax2.plot([-0.03, 0.03], [1.0, 1.08], "k-", linewidth=2)   # nose
for ch in SELECTED_CHANNELS:
    if ch in topo_pos and ch in ch_shap:
        x, y_pos = topo_pos[ch]
        v = ch_shap[ch] / max_val
        ax2.scatter(x, y_pos, s=600*v + 50, c=[[v, 0.1, 0.1]],
                    alpha=0.85, zorder=5)
        ax2.text(x, y_pos - 0.12, ch, ha="center", fontsize=8, fontweight="bold")
ax2.set_xlim(-1.15, 1.15)
ax2.set_ylim(-1.15, 1.25)
ax2.set_aspect("equal")
ax2.axis("off")
ax2.set_title("Topographic SHAP Map\n(bubble size ∝ importance)", fontsize=12)

plt.suptitle("EEG Channel Contributions — SHAP Analysis", fontsize=14)
plt.tight_layout()
plt.savefig(SHAP_DIR / "shap_channel_topo.png", dpi=150, bbox_inches="tight")
plt.show()
print(ch_df.to_string(index=False))

# %%
# ============================================================
# 16. Interactive HTML Force Plot
#     Open the saved .html file in any browser.
# ============================================================

try:
    force_html = shap.force_plot(
        expected_value,
        shap_values,
        X_explain,
        feature_names=feature_names.tolist(),
        show=False,
    )
    shap.save_html(str(SHAP_DIR / "shap_force_plot.html"), force_html)
    print(f"Interactive force plot saved → {SHAP_DIR / 'shap_force_plot.html'}")
    print("Open that file in a browser to explore individual predictions.")
except Exception as e:
    print(f"Force plot skipped: {e}")

# %%
# ============================================================
# 17. Summary Statistics & Directionality Table
# ============================================================

mean_shap  = shap_values.mean(axis=0)
std_shap   = shap_values.std(axis=0)
pct_pos    = (shap_values > 0).mean(axis=0) * 100   # % epochs where feature ↑ P(PD)

summary_stats = pd.DataFrame({
    "feature":               feature_names,
    "mean_shap":             mean_shap,
    "std_shap":              std_shap,
    "mean_abs_shap":         mean_abs_shap,
    "pct_epochs_pushes_PD":  pct_pos,
    "direction":             np.where(mean_shap > 0, "PD_positive", "CTL_positive"),
}).sort_values("mean_abs_shap", ascending=False)

summary_stats.to_csv(SHAP_DIR / "shap_summary_statistics.csv", index=False)

pd_drivers  = summary_stats[summary_stats["direction"] == "PD_positive"].head(10)
ctl_drivers = summary_stats[summary_stats["direction"] == "CTL_positive"].head(10)

print("==== TOP 10 FEATURES DRIVING PD OFF PREDICTION ====")
print(pd_drivers[["feature","mean_shap","mean_abs_shap","pct_epochs_pushes_PD"]].round(5).to_string(index=False))

print("\n==== TOP 10 FEATURES DRIVING CONTROL PREDICTION ====")
print(ctl_drivers[["feature","mean_shap","mean_abs_shap","pct_epochs_pushes_PD"]].round(5).to_string(index=False))

# %%
# ============================================================
# 18. Final Output Summary
# ============================================================

files = sorted(SHAP_DIR.iterdir())
print(f"All SHAP outputs saved to: {SHAP_DIR}")
print(f"Files written: {len(files)}\n")
for f in files:
    print(f"  {f.name:<48}  {f.stat().st_size / 1024:>7.1f} KB")

print()
print("=" * 52)
print("  SHAP EXPLAINABILITY REPORT COMPLETE")
print("=" * 52)
print(f"  Model baseline P(PD_OFF): {expected_value:.4f}")
print(f"  Epochs explained        : {n_explain}")
print(f"  Top feature (|SHAP|)    : {importance_df['feature'].iloc[0]}")
print(f"  Top channel             : {ch_df['channel'].iloc[0]}")
print(f"  Top feature group       : {band_df['feature_group'].iloc[0]}")
