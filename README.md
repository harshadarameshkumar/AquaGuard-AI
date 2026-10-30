# AquaGuard AI: Deep Neural Network-Based Water Quality Prediction and Contamination Risk Classification

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Academic Status](https://img.shields.io/badge/Academic-Third--Year%20Engineering%20Capstone-success.svg)](#)
[![XAI Enabled](https://img.shields.io/badge/Explainable%20AI-SHAP%20%2F%20Permutation-purple.svg)](#)

---

## 1. Project Overview
**AquaGuard AI** is an academic Deep Learning research project and software framework engineered to evaluate drinking water potability and classify contamination severity using continuous physicochemical telemetry.

Ingesting nine primary physicochemical parameters (pH, Hardness, Total Dissolved Solids, Chloramines, Sulfate, Electrical Conductivity, Total Organic Carbon, Trihalomethanes, and Turbidity), AquaGuard AI enforces a **leakage-free** preprocessing pipeline and deploys a Deep Feedforward Neural Network (DNN) regularized with Batch Normalization and Dropout. The system features an Explainable AI (XAI) attribution engine and automated regulatory compliance auditing cross-referenced with **World Health Organization (WHO)** and **United States Environmental Protection Agency (US-EPA)** drinking water standards.

---

## 2. Problem Statement
Access to potable water is directly threatened by industrial discharge, agricultural contamination, and decaying water distribution grids. Traditional environmental testing suffers from:
1. **Critical Latency:** Physical grab sampling and laboratory incubation introduce 24–72 hour delays before microbial or chemical contamination is detected.
2. **Non-Linear Chemical Interdependence:** Water contamination is not governed by isolated linear thresholds; complex synergies (e.g., pH shifts accelerating lead leaching or organic carbon reacting with chlorine to generate carcinogenic trihalomethanes) escape simplistic heuristic rules.
3. **Pervasive Data Leakage in Literature:** Numerous baseline studies calculate global mean imputation and normalization across entire datasets prior to train-test splitting, artificially inflating evaluation metrics.

---

## 3. Key Features
- **Leakage-Free Statistical Pipeline:** Stratified train (70%), validation (15%), and holdout test (15%) partitions strictly isolate all parameter transformations.
- **Deep Feedforward Architecture:** 3-stage dense feature representation (128 → 64 → 32) enhanced with Batch Normalization, Dropout (0.3 / 0.2), and L2 weight decay.
- **Dynamic Training Callbacks:** Implements EarlyStopping with weight restoration and adaptive learning rate decay (`ReduceLROnPlateau`).
- **Comprehensive Benchmarking:** Direct comparative performance analysis against ensemble baselines (Random Forest and Gradient Boosting).
- **Explainable AI (XAI):** Permutation feature attribution diagnosing parameters driving model risk decisions.
- **Safety-First Inference Engine:** Audits inputs against physical limits and WHO/EPA standards, generates multi-tier risk classifications (`LOW_RISK`, `MODERATE_RISK`, `HIGH_CONTAMINATION_RISK`), and outputs explicit legal/academic disclaimers.
- **Automated Academic PDF Report Generator:** Compiles a 26-section university-grade PDF report (`report/DNN_Project_Report.pdf`) with embedded visualizations and genuine references.

---

## 4. Technologies Used
- **Language:** Python 3.10+
- **Deep Learning Framework:** TensorFlow 2.16+ / Keras 3
- **Data Science & ML:** NumPy, Pandas, Scikit-learn, SciPy, Joblib
- **Visualization:** Matplotlib, Seaborn
- **Interactive Computing:** Jupyter Notebook
- **Document Generation:** ReportLab
- **Version Control:** Git & GitHub

---

## 5. Deep Neural Network Architecture
The AquaGuard AI model topology consists of:
```
Input Vector [9 Physicochemical Features]
   │
   ▼
Dense Layer (128 Units, ReLU Activation, L2=1e-4 Regularization)
   │
   ▼
Batch Normalization Layer (gamma, beta learnable parameters)
   │
   ▼
Dropout Layer (Rate = 0.30)
   │
   ▼
Dense Layer (64 Units, ReLU Activation, L2=1e-4 Regularization)
   │
   ▼
Dropout Layer (Rate = 0.20)
   │
   ▼
Dense Layer (32 Units, ReLU Activation)
   │
   ▼
Output Layer (1 Neuron, Sigmoid Activation) ──► Potability Probability [0.0 - 1.0]
```

### Mathematical Formulation
$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$
$$\mathbf{a}^{[l]} = \text{ReLU}(\mathbf{z}^{[l]}) = \max(0, \mathbf{z}^{[l]})$$
$$\hat{y} = \sigma(\mathbf{z}^{[L]}) = \frac{1}{1 + e^{-\mathbf{z}^{[L]}}}$$
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right] + \frac{\lambda}{2N} \sum_{l=1}^L \|\mathbf{W}^{[l]}\|_F^2$$

---

## 6. Dataset Information & Source
The study employs the 3,276-sample **Water Potability Benchmark Dataset**, tracking nine parameters:
- **ph:** Dimensionless (WHO permissible: 6.5–8.5)
- **Hardness:** mg/L (desirable: < 300 mg/L)
- **Solids (TDS):** ppm (WHO limit: < 1000 ppm)
- **Chloramines:** ppm (EPA MRDL: < 4.0 ppm)
- **Sulfate:** mg/L (WHO secondary standard: < 250 mg/L)
- **Conductivity:** μS/cm (standard guide: < 400 μS/cm)
- **Organic Carbon (TOC):** ppm (alert threshold: < 2.0 ppm)
- **Trihalomethanes (THMs):** μg/L (EPA MCL: < 80.0 μg/L)
- **Turbidity:** NTU (WHO benchmark: < 5.0 NTU)

*Target:* `Potability` (0 = Non-potable / elevated risk, 1 = Potable).  
*License:* Open Database License (ODbL) / Public Academic Domain.

---

## 7. Repository Structure
```
AquaGuard-AI/
│
├── README.md                           # Master Project Documentation
├── requirements.txt                    # Python Dependencies
├── .gitignore                          # Standard Python / ML Exclusions
│
├── report/
│   └── DNN_Project_Report.pdf          # Full 26-Section Academic Report
│
├── dataset/
│   ├── README.md                       # Dataset Dictionary & Integrity Notes
│   ├── water_potability.csv            # Benchmark Dataset
│   └── processed_splits.npz            # Leakage-Free Preprocessed Train/Val/Test Arrays
│
├── src/
│   ├── fetch_data.py                   # Dataset Downloader & Fallback Loader
│   ├── preprocess.py                   # Preprocessing & Leakage-Free Scaling
│   ├── train.py                        # TensorFlow DNN & Baseline Training Pipeline
│   ├── evaluate.py                     # Evaluation, ROC, Confusion Matrix & XAI
│   ├── predict.py                      # Production Inference & Safety Audit Engine
│   ├── generate_notebook.py            # Academic Jupyter Notebook Generator
│   └── generate_report.py              # ReportLab Academic PDF Generator
│
├── notebooks/
│   └── AquaGuard_Implementation.ipynb  # Step-by-Step Educational Notebook
│
├── models/
│   ├── aquaguard_dnn_best.keras        # Best Checkpointed TensorFlow Keras Model
│   ├── aquaguard_dnn_final.keras       # Final Trained Epoch Weights
│   ├── baseline_rf.joblib              # Random Forest Model Artifact
│   ├── baseline_gb.joblib              # Gradient Boosting Artifact
│   └── preprocessor.joblib             # Scaler & Median Imputer Objects
│
├── results/
│   ├── accuracy.png                    # Accuracy Progression Over Epochs
│   ├── loss.png                        # Loss Convergence Curves
│   ├── confusion_matrix.png            # Normalized Confusion Matrix Heatmap
│   ├── roc_curve.png                   # Comparative ROC-AUC Curves
│   ├── model_comparison.png            # Bar Chart Comparison Across Models
│   ├── feature_importance.png          # XAI Parameter Attribution Rankings
│   ├── correlation_heatmap.png         # Parameter Correlation Matrix
│   ├── feature_distributions.png       # Raw Feature Distributions
│   └── metrics_summary.json            # Machine-Readable Metric Logs
│
└── docs/
    └── methodology.md                  # Comprehensive Mathematical Documentation
```

---

## 8. Installation & Execution Guide

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/AquaGuard-AI/AquaGuard-AI.git
cd AquaGuard-AI
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Full Pipeline
```bash
# 1. Download & preprocess dataset
python src/fetch_data.py
python src/preprocess.py

# 2. Train Deep Neural Network and baseline models
python src/train.py

# 3. Evaluate models and generate figures
python src/evaluate.py

# 4. Generate the Jupyter Notebook and Academic PDF Report
python src/generate_notebook.py
python src/generate_report.py

# 5. Run real-time test inference
python src/predict.py
```

---

## 9. Experimental Results & Visualizations

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AquaGuard DNN (Config 1: 128-64-32)** | **0.8841** | **0.6250** | **0.5893** | **0.6066** | **0.6582** |
| AquaGuard DNN (Config 2: 256-128-64-16) | 0.8862 | 0.5833 | 0.5000 | 0.5385 | 0.6350 |
| Random Forest (150 trees, balanced) | 0.8862 | 0.6000 | 0.5357 | 0.5660 | 0.6420 |
| Gradient Boosting (100 estimators) | 0.8821 | 0.5500 | 0.4821 | 0.5140 | 0.6433 |

*Key Findings:* The regularized Deep Neural Network achieves superior recall and balanced F1-score while preserving low false-alarm rates. Early stopping and dropout effectively prevent overfitting on noisy sensor values.

---

## 10. Explainable AI (XAI) Insights
Permutation feature attribution indicates that:
1. **pH** and **Chloramines** exert the strongest influence on model classifications.
2. **Trihalomethanes** and **Turbidity** provide critical secondary hazard flags.
3. Parameters like **Conductivity** and **Hardness** modulate baseline mineral acceptability.

---

## 11. Important Academic & Safety Limitations
> **CRITICAL LEGAL NOTICE:**  
> This system models physicochemical parameters only. It **does NOT** detect:
> - Biological pathogens (e.g., *Cryptosporidium*, *Giardia lamblia*, *Vibrio cholerae*, *Escherichia coli*).
> - Dissolved heavy metals (Lead, Arsenic, Cadmium, Hexavalent Chromium).
> 
> Outputs from AquaGuard AI are computational risk classifications intended for operational decision support and research exploration. They **do not constitute certified municipal drinking water safety endorsements**.

---

## 12. Future Scope
1. **Edge Deployment:** Porting model weights to TensorFlow Lite for deployment on low-cost ESP32/Raspberry Pi environmental sensing nodes.
2. **Multi-Spectral Integration:** Incorporating UV-Vis fluorescence sensors for real-time bacterial presence estimation.
3. **Temporal Dynamics:** Integrating Recurrent Neural Networks (LSTM/GRU) or Temporal Convolutional Networks (TCN) to model seasonal watershed rainfall cycles.
4. **Federated Multi-Plant Learning:** Enabling inter-utility collaborative training without centralizing raw reservoir data.
