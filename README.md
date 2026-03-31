# 🧠 EEG-Based AI System for Parkinson's Disease Analysis

An end-to-end **Artificial Intelligence (AI) pipeline** for EEG signal analysis, developed as part of the  
**COS6032-E – Industry AI Project**.  
This project explores the use of **machine learning and deep learning techniques** to extract EEG-based
biomarkers relevant to **Parkinson's disease (PD)**, with a strong emphasis on **technical rigour,
reproducibility, and responsible AI practice**.

---

## 📌 Project Overview

Parkinson's disease is a progressive neurological disorder associated with abnormal neural oscillations
observable through electroencephalography (EEG). This project designs and implements a **modular EEG
analysis pipeline** that processes raw EEG signals from a three-stimulus auditory oddball paradigm,
extracts meaningful features, and evaluates multiple AI models to support research into cognitive
impairments in PD — specifically the neural correlates of attention and habituation.

The pipeline distinguishes between **PD patients (ON and OFF medication)** and **healthy controls (CTL)**,
enabling analysis of dopaminergic medication effects on EEG biomarkers.

The system follows the **CDIO (Conceive–Design–Implement–Operate)** framework and reflects industry-style
AI development practices within a healthcare research context.

---

## 🎯 Aims and Objectives

### Aim
To design and implement a robust AI-driven EEG analysis system capable of identifying informative neural
patterns related to Parkinson's disease using a three-stimulus auditory oddball task.

### Objectives
- Preprocess raw BrainVision EEG recordings using established signal-processing techniques (filtering,
  ICA, bad channel rejection, re-referencing)
- Extract **Event-Related Potentials (ERPs)** and **Time-Frequency** features (power, ITPC) for
  Standard, Target, and Novelty stimuli
- Classify **PD-ON vs CTL**, **PD-OFF vs CTL**, and **PD-ON vs PD-OFF** using FFT-based SVM and
  deep learning (EEGNet)
- Identify spatially and spectrally informative features via decremental feature selection
- Ensure ethical, transparent, and responsible AI development
- Provide clear documentation and reproducible workflows

---

## 🗂️ Project Management Methodology

### CDIO Framework

- **Conceive** — Problem definition, domain research, ethical analysis, and definition of aims and constraints
- **Design** — End-to-end EEG pipeline design including preprocessing strategy, feature extraction
  (ERP, TF, FFT), model selection (EEGNet, SVM), and evaluation planning
- **Implement** — Development of preprocessing, feature extraction, and modelling code with iterative
  experimentation and continuous validation
- **Operate** — Demonstration of a working AI prototype, interpretation of results in a research
  context, and reflection on limitations

### Agile Practices
- Iterative and incremental development driven by learning curve analysis
- Regular reflection and refinement of hyperparameters and regularisation strategy
- Transparent version control via GitHub

---

## 📊 Dataset Description

### Dataset
**OpenNeuro ds003490** — *EEG – 3-Stimulus Auditory Oddball and Rest in Parkinson's Disease*

### Key Characteristics
| Property | Detail |
|---|---|
| Modality | Scalp EEG (64 channels, BrainVision) |
| Paradigm | 3-stimulus auditory oddball task |
| Stimuli | Standard (S201), Target (S200), Novelty (S202) |
| Participants | 50 total: 25 PD patients, 25 healthy controls |
| PD sessions | 2 sessions per patient: medication ON and OFF |
| CTL sessions | 1 session per control |
| Sampling rate | 500 Hz (downsampled to 100 Hz for classification) |
| Reference | CPz (re-referenced to average) |

### Subject Groups and Comparisons
- **PD-OFF vs CTL** — effect of PD without medication

---

## 🔬 System Pipeline

### Step 1 — Preprocessing (`preprocessing.py`)
- Load BrainVision `.vhdr` files via MNE-BIDS
- Force EEG channel type assignment for 10-20 sites
- Bad channel detection (flat threshold: 1 µV; noisy z-score > 3)
- Spherical spline interpolation of bad channels
- Average re-referencing (CPz recovered before re-ref)
- Bandpass filter: 0.5–40 Hz; notch filter: 60 Hz (+ 120 Hz harmonic)
- Resample to **100 Hz** (matching MATLAB Step 4)
- ICA: fitted on 1 Hz high-pass copy; blink/heartbeat components
  removed via Fp1/Fp2 proxy
- Event-locked epoching (−2 to +2 s); amplitude rejection (150 µV)
- **Output:** per-subject, per-session `.fif` files named
  `sub-XXX_ON-epo.fif`, `sub-XXX_OFF-epo.fif`, `sub-XXX_CTL-epo.fif`
- **SINGLETRIAL_ODDBALL.npz** built for classification:
  condition-specific windows (−250 to +1250 ms), per-condition baseline
  correction, 30 random Standards selected, trinary trial splits (first /
  middle / last third), trial-count matching across conditions

### Step 2 — ERP & Time-Frequency Analysis
- ERPs computed for Standard, Target, and Novelty conditions
- Wavelet convolution (50 log-spaced frequencies, 1–50 Hz)
- Morlet wavelet power (dB baseline-corrected) and ITPC computed
- Trinary split ERPs for habituation analysis (first vs last third of trials)
- QC plots: channel SD bar chart, PSD before/after filtering, ICA overlay

### Step 3 — Classification (`train.py`)

#### FFT + SVM Classifier (MATLAB Step 5 equivalent)
- FFT power extracted from 250–1000 ms window per trial and channel
- Feature vector: channels × frequencies (flattened)
- Linear SVM with StandardScaler, stratified 5-fold CV per subject pair
- Comparisons: ON_CTL, OFF_CTL, ON_OFF across Standard, Target, Novelty

#### Decremental Feature Importance (MATLAB Step 8 equivalent)
- 50 permutations of greedy feature elimination
- Features retained only if removal keeps accuracy ≥ 95% of full-set accuracy
- Results visualised by frequency band (Delta, Theta, Alpha, Beta, Gamma)

#### EEGNet Deep Learning Classifier
- EEGNet architecture (Lawhern et al. 2018): F1=4, D=2, F2=8
- Input: (batch, 1, channels, timepoints)
- Training on PD-ON vs CTL (configurable via `comparison` parameter)
- **Regularisation strategy:**
  - Dropout: 0.70
  - Weight decay: 1e-3
  - Gaussian noise augmentation (σ=0.05, train only)
  - Channel dropout (p=0.05, train only)
  - EEG Mixup (α=0.2): interpolates epoch pairs to create virtual subjects
  - Label smoothing: 0.05
- **Normalisation:** subject-wise z-score applied before CV split
- **Optimiser:** Adam (lr=3e-4); ReduceLROnPlateau scheduler
- **Early stopping:** patience=25, min_epochs=20 grace period
- **CV:** 5-fold StratifiedGroupKFold (no subject in both train and val)

### Step 4 — Evaluation
- Epoch-level and subject-level metrics: Accuracy, AUC-ROC, Balanced Accuracy
- Confusion matrix (subject-level)
- Learning curve plots (train/val NLL loss per fold)
- CV summary table with mean ± SD across folds

---

## 📂 Repository Structure

```
EEG-PD-Analysis/
│
├── preprocessing.py          # Steps 1–2: preprocessing + singletrial dataset
├── train.py                  # Steps 3–4: FFT/SVM + EEGNet classification
│
├── data/
│   └── ds003490/             # BIDS dataset (not included in repo)
│       ├── participants.tsv
│       └── sub-XXX/
│
├── Signal Code/Outputs/
│   ├── cache_epochs_binary/  # Per-session .fif epoch files
│   ├── qc_plots_binary/      # QC plots per subject
│   ├── eegnet_model/         # Model checkpoints, CV results, plots
│   ├── SINGLETRIAL_ODDBALL.npz
│   └── preprocessing_summary.tsv
│
└── README.md
```

---

## ⚖️ Ethical and Responsible AI Considerations

- **Data Privacy** — All EEG data is fully anonymised; no personally identifiable information is used
- **Transparency** — All preprocessing decisions (ICA component selection, bad channel thresholds,
  epoch rejection criteria) are logged and reproducible
- **Clinical Responsibility** — The system is intended for **research and decision-support only**;
  no automated diagnosis or clinical claims are made
- **Bias and Fairness** — Balanced subject counts (25 PD, 25 CTL); subject-wise normalisation prevents
  amplitude-based shortcuts; StratifiedGroupKFold prevents subject leakage across folds
- **Data Leakage Prevention** — Subject-wise z-score applied before CV; no preprocessing statistics
  fitted on validation data; group-aware CV splitting
- **Hyperparameter Transparency** — All hyperparameters documented with rationale; learning curves
  reported to evidence training stability

---

## 🛠️ Technologies Used

| Category | Library |
|---|---|
| EEG processing | MNE-Python, MNE-BIDS, pybids |
| Numerics | NumPy, SciPy, pandas |
| Deep learning | PyTorch |
| Classical ML | scikit-learn (SVM, StratifiedGroupKFold) |
| Visualisation | Matplotlib, Seaborn |
| Version control | GitHub |

### Installation

```bash
pip install mne mne-bids pybids torch torchvision scikit-learn \
            tqdm matplotlib seaborn pandas numpy
```

### Running the Pipeline

```bash
# Step 1: Preprocess all subjects and build classification dataset
python preprocessing.py

# Step 2: Train and evaluate classifier
python train.py
```

> **Before running**, update the paths at the top of `preprocessing.py`:
> ```python
> BIDS_ROOT = Path("/path/to/ds003490")
> OUT_DIR   = Path("/path/to/outputs")
> ```

---

## 📈 Key Results

Evaluation uses **cross-validation only** (training-set evaluation figures are not reported as valid performance estimates).

| Metric | Epoch-level | Subject-level |
|---|---|---|
| Mean CV Accuracy | reported in `cv_results.tsv` | reported in `cv_results.tsv` |
| Mean CV AUC-ROC | reported in `cv_results.tsv` | reported in `cv_results.tsv` |
| Mean CV Balanced Acc | reported in `cv_results.tsv` | reported in `cv_results.tsv` |

---

## 🎓 Intended Use

This project is intended for academic and educational purposes, EEG-based biomarker research, and
demonstration of applied AI skills in healthcare contexts.

It is **not intended for direct clinical deployment** without further clinical validation and regulatory approval.

---

## 🔮 Future Work
- Nested cross-validation to produce fully unbiased performance estimates
- Integration of resting-state EEG features alongside oddball task features
- Multimodal fusion with clinical scores (UPDRS, years since diagnosis)
- Improved model interpretability via saliency maps and SHAP values
- Cross-dataset validation to assess generalisability
- Exploration of transformer-based EEG architectures

---

## 🤖 AI Use Disclosure

Generative AI tools were used **only as a supporting aid** for code assistance and documentation refinement.  
All technical decisions, implementations, analyses, and conclusions were developed by the project author(s).
