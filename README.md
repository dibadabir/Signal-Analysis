# 🧠 EEG-Based AI System for Parkinson's Disease Analysis

An end-to-end Artificial Intelligence (AI) pipeline for EEG signal analysis, developed as part of the  
COS6032-E – Industry AI Project.  
This project explores the use of machine learning techniques to extract EEG-based biomarkers relevant
to Parkinson's disease (PD), with a strong emphasis on **technical rigour, reproducibility, and
responsible AI practice**.

---

## 📌 Project Overview

Parkinson's disease is a progressive neurological disorder associated with abnormal neural oscillations
observable through electroencephalography (EEG). This project designs and implements a **modular EEG
analysis pipeline** that processes raw EEG signals from a three-stimulus auditory oddball paradigm,
extracts meaningful features, and evaluates a Support Vector Machine (SVM) model to support research
into cognitive impairments in PD.

The pipeline focuses on classifying PD patients (OFF medication) against healthy controls (CTL),
enabling analysis of unmedicated Parkinson's disease effects on EEG biomarkers.

The system follows the CDIO (Conceive–Design–Implement–Operate) framework and reflects industry-style
AI development practices within a healthcare research context.

---

## 🎯 Aims and Objectives

### Aim
To design and implement a robust AI-driven EEG analysis system capable of identifying informative neural
patterns related to Parkinson's disease using a three-stimulus auditory oddball task.

### Objectives
- Preprocess raw BrainVision EEG recordings using established signal-processing techniques (filtering,
  ICA, bad channel rejection, re-referencing)
- Extract Event-Related Potential (ERP) and spectral bandpower features from target-locked epochs
- Classify PD-OFF vs CTL using a spectral + ERP SVM with full cross-validation and threshold optimisation
- Identify spatially and spectrally informative features via SHAP value analysis
- Ensure ethical, transparent, and responsible AI development
- Provide clear documentation and reproducible workflows

---

## 🗂️ Project Management Methodology

### CDIO Framework

<img width="660" height="422" alt="Screenshot 2026-04-14 at 16 44 08" src="https://github.com/user-attachments/assets/77cb1e9b-a107-4a7b-b0f7-7bef328f5799" />


### Agile Practices
- Iterative and incremental development across multiple notebook versions
- Regular reflection and refinement of feature sets, hyperparameters, and regularisation strategy
- Transparent version control via GitHub

- ## 👥 Team

| Name | Role |
|---|---|
| Justyna Dobersztajn | App interface & design, project management |
| Aleeza Azad | Technical documentation & model development|
| Diba Dabiransari | Model development & ethical analysis|

---

## 📊 Dataset Description

### Dataset
[OpenNeuro ds003490](https://openneuro.org/datasets/ds003490/versions/1.1.0)

### Key Characteristics
| Property | Detail |
|---|---|
| Modality | Scalp EEG (64 channels, BrainVision) |
| Paradigm | 3-stimulus auditory oddball task |
| Stimuli | Standard (S201), Target (S200), Novelty (S202) |
| Participants | 50 total: 25 PD patients, 25 healthy controls |
| PD sessions | 2 sessions per patient: medication ON and OFF |
| CTL sessions | 1 session per control |
| Original sampling rate | 500 Hz |
| Resampled to | 250 Hz (post-preprocessing) |
| Reference | CPz (re-referenced to average) |

### Subject Groups and Comparisons
- PD-OFF vs CTL — Classification target: effect of unmedicated Parkinson's disease on EEG biomarkers

---

## 🔬 System Pipeline

### Step 1 — Preprocessing

- Load BrainVision `.vhdr` files via MNE-BIDS on Google Colab
- Focus on a 16-channel subset for cleaner classical ML features:
  `Fp1, Fp2, F3, F4, Fz, FC1, FC2, FCz, C3, C4, Cz, P3, P4, Pz, T7, T8`
- Bad channel detection: flat threshold ≤ 1 µV; noisy z-score > 3.0
  (Fp1/Fp2 protected as EOG proxies; max 30% of channels may be marked bad)
- Spherical spline interpolation of bad channels
- Average re-referencing (CPz recovered before re-reference)
- Bandpass filter: 0.5–40 Hz; notch filter: 50 Hz (UK mains frequency)
- Resample to 250 Hz
- ICA: fitted on 1 Hz high-pass copy (`n_components=0.99` variance); blink/heartbeat
  components removed using Fp1/Fp2 as EOG proxies
- Target-only epoching: event code 200 (S200), window −200 to +1000 ms,
  baseline correction (−200 to 0 ms), amplitude rejection at 150 µV peak-to-peak
- Output: per-subject, per-session `.fif` files:
  `sub-XXX_OFF-epo.fif`, `sub-XXX_CTL-epo.fif`

### Step 2 — Feature Extraction

Features are extracted from a 9-channel clinical subset: `F3, F4, Fz, C3, C4, Cz, P3, P4, Pz`

For each epoch, the following feature groups are computed:

- Absolute bandpower — Welch PSD mean power per channel × band
- Relative bandpower — Band power normalised by total power per channel × band
- Band ratios — Per channel: theta/beta, alpha/theta, alpha/beta
- ERP window means — Mean amplitude per channel in three post-stimulus windows:
  250–400 ms, 400–700 ms, 700–1000 ms
- Global summaries — Scalp-average relative power for delta, theta, alpha, beta

Frequency bands used:

| Band | Range |
|---|---|
| Delta | 1–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–12 Hz |
| Beta | 12–30 Hz |

### Step 3 — SVM Classification

#### Data Splitting (Subject-Aware)
- 80% train/val — 20% held-out test via `GroupShuffleSplit` (no subject overlap)
- Within train/val: 75% train — 25% validation via a second `GroupShuffleSplit`
- Training set balanced by undersampling to equal class counts; cap of 60 epochs per subject

#### Hyperparameter Optimisation
- `GridSearchCV` with inner `GroupKFold` (5 folds, subject-aware)
- Grid: `C` ∈ {0.5, 1, 5, 10, 20}; `gamma` ∈ {scale, 0.01, 0.001};
  `class_weight` ∈ {1:1.5, 1:2.0, 1:3.0, balanced}
- Scoring: ROC-AUC

#### Threshold Optimisation
- Decision thresholds searched over [0.35, 0.65] for both epoch-level and subject-level predictions
- Subject-level predictions: mean predicted probability averaged across a subject's epochs

#### Fold-by-Fold Validation
- Re-trains using best hyperparameters across 5 GroupKFold splits of the training set
- Reports per-fold: accuracy, precision, recall, F1, ROC-AUC at both epoch and subject level

### Step 4 — SHAP Interpretability

SHAP values are computed on the trained SVM to explain feature contributions:

- Top 25 features by SHAP importance — `P4_beta_rel` is the most discriminative single feature,
  followed by frontal delta/theta/beta relative power at F3 and Fz
- Feature group contributions — alpha/beta ratio (19.3%) and beta relative power (15.6%)
  collectively account for over a third of total SHAP importance
- Channel-level SHAP map — P4, Fz, and F3 carry the highest aggregate importance;
  visualised as a topographic bubble map

> SHAP results indicate the SVM relies primarily on **relative beta and delta power at posterior
> and frontal sites**, with cross-frequency ratios (alpha/beta, alpha/theta) providing additional
> discriminative signal — consistent with known PD-related oscillatory abnormalities.

### Step 5 — Evaluation & Outputs

- Epoch-level and subject-level metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Fold-by-fold metric plots (train/val per fold)
- Confusion matrices and ROC curves (epoch- and subject-level)
- All results exported to Excel (`svm_all_results.xlsx`) and CSV
- Trained model saved via `joblib` (`best_svm_model.joblib`)

---

## ⚖️ Ethical and Responsible AI Considerations

The project was evaluated against six responsible AI dimensions, achieving an average score of 7.77 / 10:

![Ethical AI Radar Chart](https://github.com/dibadabir/Signal-Analysis/blob/main/Diagrams/Ethic%20report.png)

Key practices underpinning these scores:

- Data Privacy — All EEG data is fully anonymised; no personally identifiable information is used
- Transparency — All preprocessing decisions (ICA thresholds, bad channel detection,
  epoch rejection criteria) are logged and reproducible; SHAP values provide post-hoc explanations
- Clinical Responsibility — The system is intended for research and decision-support only;
  no automated diagnosis or clinical claims are made
- Bias & Fairness — Balanced subject counts (25 PD, 25 CTL); subject-wise normalisation prevents
  amplitude-based shortcuts; GroupShuffleSplit and GroupKFold prevent subject leakage across splits
- Data Leakage Prevention — StandardScaler fitted on training fold only; feature selection and PCA
  fitted within each fold; group-aware splitting throughout
- Hyperparameter Transparency — All hyperparameters documented with rationale; fold-wise metrics
  reported to evidence performance stability

---

## 🚀 Deployment — Streamlit Prototype

The trained SVM is deployed as a clinician-facing web application built with Streamlit and hosted
on Google Colab via an ngrok tunnel. The app exposes the full preprocessing-to-prediction pipeline
so that a raw EEG recording can be uploaded and classified without any local environment setup.

### App Architecture

The app (`app.py`) is written as a single-file Streamlit application. Navigation is provided by a
sidebar with three pages:

1 — Diagnosis Tool
The core clinical page. A clinician uploads the patient's EEG files, the embedded pipeline
preprocesses them, extracts features, and the loaded SVM returns a subject-level classification.

- Upload accepts EEGLAB `.set` + optional `.fdt` binary data file + optional `events.tsv`
- The full preprocessing chain runs in-app, matching the training pipeline exactly:
  channel standardisation → bad channel detection & interpolation → average re-reference →
  bandpass/notch filter → resample to 250 Hz → ICA artefact removal → target epoch extraction
- Event detection supports both MNE annotations and trigger channels; the `events.tsv` sidecar
  is parsed for a "Target Tone" hint to robustly identify the S200 event code
- Subject-level prediction: epoch probabilities are averaged and compared against a **fixed
  decision threshold of 0.51**
- Confidence is banded into three tiers — High (≥ 85%), Moderate (≥ 70%), Low (< 70%)
- Results are displayed in a colour-coded card (red for Parkinson's, blue for Control) with
  plain-language phrasing and a clear disclaimer that the output supports — but does not replace —
  clinical judgement
- An expandable Quality checks panel shows: available/missing channels, interpolated bad
  channels, ICA components removed, detected event annotation key and code, feature count, and
  a per-epoch probability table with CSV download

2 — Ethical AI
Summarises the EthiCheck AI Ethics Assessment results (overall grade: Good; risk profile:
Medium) and offers a download button for the full PDF report.

### Running the App (Google Colab)

Prerequisites: `best_svm_model.joblib` must be saved to Google Drive and the `MODEL_PATH`
constant in `app.py` updated to point to it before launching.

```python
# Cell 1 — kill any existing processes
!pkill -f streamlit
!pkill -f ngrok

# Cell 2 — install pinned dependencies
!pip -q install streamlit==1.44.1 mne==1.10.1 numpy==2.2.4 pandas==2.2.3 \
                scipy==1.15.2 matplotlib==3.10.1 joblib==1.4.2 \
                scikit-learn==1.6.1 pyngrok==7.2.5

# Cell 3 — write app.py to /content/app.py  (%%writefile cell in notebook)

# Cell 4 — authenticate ngrok and launch
from pyngrok import ngrok

ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")   # replace with your token
ngrok.kill()

get_ipython().system_raw("streamlit run /content/app.py --server.port 8501 &")
public_url = ngrok.connect(8501)
print(public_url)   # open this URL to access the app
```

> Note: Replace `YOUR_NGROK_AUTH_TOKEN` with your personal token from
> [ngrok.com](https://ngrok.com). The ngrok token in the repository notebook must be rotated
> before sharing publicly.

### Input File Format

| File | Required | Description |
|---|---|---|
| `.set` | ✅ Yes | EEGLAB dataset file (contains header + optionally inline data) |
| `.fdt` | Recommended | Binary data file required if `.set` does not embed the EEG data |
| `events.tsv` | Optional | BIDS-format events sidecar; improves target event detection |

---

## 🛠️ Technologies Used

| Category | Library / Tool |
|---|---|
| EEG processing | MNE-Python 1.10.1, MNE-BIDS, pybids |
| Numerics | NumPy 2.2.4, SciPy 1.15.2, pandas 2.2.3 |
| Classical ML | scikit-learn 1.6.1 (SVM, GroupKFold, GridSearchCV, PCA, SelectKBest) |
| Interpretability | SHAP |
| Visualisation | Matplotlib 3.10.1, Seaborn |
| Deployment | Streamlit 1.44.1, pyngrok 7.2.5 |
| Persistence | joblib 1.4.2, openpyxl |
| Environment | Google Colab (preprocessing + deployment), local Python (modelling) |
| Version control | GitHub |

### Installation

```bash
# Research pipeline
pip install mne mne-bids pybids scikit-learn shap \
            matplotlib seaborn pandas numpy openpyxl joblib tqdm

# Streamlit app (pinned versions matching Colab deployment)
pip install streamlit==1.44.1 mne==1.10.1 numpy==2.2.4 pandas==2.2.3 \
            scipy==1.15.2 matplotlib==3.10.1 joblib==1.4.2 \
            scikit-learn==1.6.1 pyngrok==7.2.5
```

### Running the Pipeline

```bash
# Step 1: Preprocess all subjects → .fif epoch files
# Run Final_Pre_processing_for_SVM.ipynb in Google Colab
# Update BIDS_ROOT and OUT_DIR paths at the top of the notebook

# Step 2: Feature extraction, SVM training, SHAP analysis
# Run 3rd_version_of_SVM.ipynb
# Update OUT_DIR path to point to the cache_epochs_binary folder

# Step 3: Deploy Streamlit app
# Run Steamlit_.ipynb in Google Colab (see Deployment section above)
```

---

## 🎓 Intended Use

This project is intended for academic and educational purposes, EEG-based biomarker research, and
demonstration of applied AI skills in healthcare contexts.

It is not intended for direct clinical deployment without further clinical validation and regulatory
approval.

---

## 🤖 AI Use Disclosure

Generative AI tools were used only as a supporting aid for code assistance and documentation
refinement. All technical decisions, implementations, analyses, and conclusions were developed by the
project author(s).
