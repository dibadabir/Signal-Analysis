# 🧠 EEG-Based AI System for Parkinson’s Disease Analysis

An end-to-end **Artificial Intelligence (AI) pipeline** for EEG signal analysis, developed as part of the  
**COS6032-E – Industry AI Project**.  
This project explores the use of **machine learning and deep learning techniques** to extract EEG-based
biomarkers relevant to **Parkinson’s disease (PD)**, with a strong emphasis on **technical rigour,
reproducibility, and responsible AI practice**.

---

## 📌 Project Overview

Parkinson’s disease is a progressive neurological disorder associated with abnormal neural oscillations
that can be observed using electroencephalography (EEG).  
This project designs and implements a **modular EEG analysis pipeline** that processes raw EEG signals,
extracts meaningful features, and evaluates multiple AI models to support research into motor and
cognitive impairments in PD.

The system follows the **CDIO (Conceive–Design–Implement–Operate)** framework and reflects industry-style
AI development practices within a healthcare research context.

---

## 🎯 Aims and Objectives

### Aim
To design and implement a robust AI-driven EEG analysis system capable of identifying informative neural
patterns related to Parkinson’s disease.

### Objectives
- Preprocess raw EEG signals using established signal-processing techniques  
- Extract informative **time-domain**, **frequency-domain**, and **time–frequency** features  
- Train and evaluate machine learning and deep learning models  
- Compare feature representations and modelling approaches  
- Ensure ethical, transparent, and responsible AI development  
- Provide clear documentation and reproducible workflows  

---

## 🗂️ Project Management Methodology

### CDIO Framework
The project lifecycle follows the CDIO approach:

- **Conceive**
  - Problem definition and domain research
  - Stakeholder identification and ethical analysis
  - Definition of aims, objectives, and constraints

- **Design**
  - End-to-end EEG pipeline design
  - Selection of preprocessing, features, and models
  - Experiment and evaluation planning

- **Implement**
  - Development of preprocessing, feature extraction, and modelling code
  - Version control and iterative experimentation
  - Continuous documentation and validation

- **Operate**
  - Demonstration of a working AI prototype
  - Interpretation of results in a research context
  - Reflection on limitations and future improvements

### Agile Practices
- Iterative and incremental development  
- Clear task allocation and responsibility boundaries  
- Regular reflection and refinement  
- Transparent version control via GitHub  

---

## 📊 Dataset Description

### Dataset Overview
This project uses the **OpenNeuro dataset: _EEG – 3-Stim Auditory Oddball and Rest in Parkinson’s Disease_**.

The dataset contains **scalp EEG recordings from individuals with Parkinson’s disease** collected during
both **resting-state** and a **three-stimulus auditory oddball task**, which is commonly used to study
attention, cognitive processing, and event-related brain responses.

### Key Characteristics
- **Modality:** Electroencephalography (EEG)  
- **Paradigms:**  
  - Resting-state EEG  
  - 3-stimulus auditory oddball task  
- **Population:** 50 participants (25 Parkinson's disease patients & 25 Control versions)

---

## 🔬 System Pipeline

### 1. Data Understanding
- EEG data organised in **BIDS-compliant format**
- Initial inspection of signal quality and variability

### 2. Preprocessing
- Band-pass and notch filtering
- Artifact handling (e.g. ICA, automated rejection)
- Epoching for consistency across participants

### 3. Feature Extraction
- **Time-domain** features (mean, variance, RMS, entropy)
- **Frequency-domain** features (power spectral density, band power)
- **Time–frequency** features (wavelet-based)
- Feature aggregation for model input

### 4. Modelling
- Traditional ML models (e.g. SVM, Random Forest)
- Deep learning architectures (e.g. CNN, CNN-LSTM, EEGNet)
- Proper train/validation/test separation

### 5. Evaluation
- Metrics: accuracy, precision, recall, F1-score
- Focus on robustness and generalisation
- Comparative analysis of feature sets and models

---

## ⚖️ Ethical and Responsible AI Considerations

- **Data Privacy**
  - All EEG data is anonymised
  - No personally identifiable information is used

- **Transparency and Explainability**
  - Interpretable features prioritised where possible
  - Model assumptions and limitations explicitly documented

- **Clinical Responsibility**
  - The system is intended for **research and decision-support only**
  - No automated diagnosis or clinical claims are made

- **Bias and Fairness**
  - Dataset limitations acknowledged
  - No overgeneralisation beyond the available data

- **Sustainability**
  - Computational efficiency considered
  - Modular design enables future extensions

---

## 📁 Repository Structure (to be changed)


---

## 🛠️ Technologies Used
- **Python**
- NumPy, SciPy
- MNE-Python
- Scikit-learn
- TensorFlow / Keras
- GitHub (version control and collaboration)

---

## 🎓 Intended Use

This project is intended for:
- Academic and educational purposes  
- EEG-based biomarker research  
- Demonstration of applied AI skills in healthcare contexts  

It is **not intended for direct clinical deployment** without further clinical validation and regulatory
approval.

---

## 🔮 Future Work
- Integration of multimodal data (EEG + clinical scores)
- Improved model interpretability and explainability
- Cross-dataset validation to assess generalisability
- Exploration of graph-based EEG representations

---

## 🤖 AI Use Disclosure
Generative AI tools were used **only as a supporting aid** for code assistance and documentation refinement.  
All technical decisions, implementations, analyses, and conclusions were developed by the project author(s).

