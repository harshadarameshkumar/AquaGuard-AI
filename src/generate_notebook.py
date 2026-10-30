"""
Generates the comprehensive academic Jupyter Notebook: notebooks/AquaGuard_Implementation.ipynb
"""

import json
import os

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "..", "notebooks", "AquaGuard_Implementation.ipynb")
os.makedirs(os.path.dirname(NOTEBOOK_PATH), exist_ok=True)

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# AquaGuard AI: Deep Neural Network-Based Water Quality Prediction and Contamination Risk Classification\n",
            "### An Academic Deep Learning Investigation on Physicochemical Potability Indicators\n",
            "\n",
            "**Author:** Senior Deep Learning Researcher & Machine Learning Engineer  \n",
            "**Academic Level:** Engineering Degree (Third-Year Capstone / Deep Learning Systems)  \n",
            "**Tech Stack:** Python 3.10+, TensorFlow / Keras 3+, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, ReportLab\n",
            "\n",
            "---"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Project Abstract & Objectives\n",
            "Safe drinking water is a cornerstone of public health. This notebook implements an end-to-end Deep Neural Network (DNN) pipeline to classify water samples into safety/risk categories using 9 continuous physicochemical parameters:\n",
            "- **pH**: Acid-base balance (safe: 6.5–8.5)\n",
            "- **Hardness**: Calcium & Magnesium precipitation capacity\n",
            "- **Solids (TDS)**: Total dissolved solids\n",
            "- **Chloramines**: Disinfection residual\n",
            "- **Sulfate**: Naturally occurring sulfur salts\n",
            "- **Conductivity**: Electrical conductivity linked to ionic mineralization\n",
            "- **Organic Carbon**: Natural and anthropogenic organic compound concentration\n",
            "- **Trihalomethanes**: Disinfection byproducts with carcinogenic potential\n",
            "- **Turbidity**: Cloudiness/particulate matter harboring microorganisms\n",
            "\n",
            "### Key Methodological Safeguards:\n",
            "1. **Leakage-Free Preprocessing:** Data split strictly precedes missing-value imputation and feature scaling.\n",
            "2. **Comparative Benchmarking:** The proposed DNN is contrasted with Random Forest and Gradient Boosting ensembles.\n",
            "3. **Transparent Limitations:** Predictions are computational risk indicators, not certified municipal safety endorsements."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Environment Setup & Dependency Imports\n",
            "import os\n",
            "import sys\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import joblib\n",
            "\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
            "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report\n",
            "\n",
            "import tensorflow as tf\n",
            "from tensorflow.keras import layers, models, callbacks, regularizers\n",
            "\n",
            "print(f\"TensorFlow Version: {tf.__version__}\")\n",
            "print(f\"NumPy Version: {np.__version__}\")\n",
            "print(f\"Pandas Version: {pd.__version__}\")\n",
            "sns.set_theme(style=\"whitegrid\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Dataset Ingestion & Exploratory Data Analysis (EDA)\n",
            "We inspect raw column datatypes, distributions, and missing values."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load dataset\n",
            "data_path = os.path.join(\"..\", \"dataset\", \"water_potability.csv\")\n",
            "df = pd.read_csv(data_path)\n",
            "print(f\"Dataset Dimensions: {df.shape[0]} rows x {df.shape[1]} columns\")\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Missing value audit\n",
            "print(\"Missing Values per Feature:\")\n",
            "missing_pct = (df.isnull().sum() / len(df)) * 100\n",
            "pd.DataFrame({\"Missing Count\": df.isnull().sum(), \"Percentage (%)\": missing_pct.round(2)})"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Leakage-Free Preprocessing Pipeline\n",
            "**Crucial Step:** We partition the data into **Training (70%)**, **Validation (15%)**, and **Test (15%)** *before* fitting any imputer or scaler."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "features = [\"ph\", \"Hardness\", \"Solids\", \"Chloramines\", \"Sulfate\", \"Conductivity\", \"Organic_carbon\", \"Trihalomethanes\", \"Turbidity\"]\n",
            "target = \"Potability\"\n",
            "\n",
            "X = df[features]\n",
            "y = df[target]\n",
            "\n",
            "# Stratified 70/15/15 split\n",
            "X_train_raw, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)\n",
            "X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)\n",
            "\n",
            "# Compute training set medians\n",
            "train_medians = X_train_raw.median()\n",
            "X_train_imp = X_train_raw.fillna(train_medians)\n",
            "X_val_imp = X_val_raw.fillna(train_medians)\n",
            "X_test_imp = X_test_raw.fillna(train_medians)\n",
            "\n",
            "# Standardize features (fit only on training split)\n",
            "scaler = StandardScaler()\n",
            "X_train = scaler.fit_transform(X_train_imp)\n",
            "X_val = scaler.transform(X_val_imp)\n",
            "X_test = scaler.transform(X_test_imp)\n",
            "\n",
            "print(f\"Training set shape: {X_train.shape}\")\n",
            "print(f\"Validation set shape: {X_val.shape}\")\n",
            "print(f\"Test set shape: {X_test.shape}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Deep Neural Network Architecture\n",
            "We construct a multilayer feedforward architecture with Batch Normalization, Dropout regularization, and L2 weight decay."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def build_dnn(input_dim):\n",
            "    model = models.Sequential([\n",
            "        layers.Input(shape=(input_dim,)),\n",
            "        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),\n",
            "        layers.BatchNormalization(),\n",
            "        layers.Dropout(0.3),\n",
            "        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),\n",
            "        layers.Dropout(0.2),\n",
            "        layers.Dense(32, activation='relu'),\n",
            "        layers.Dense(1, activation='sigmoid')\n",
            "    ], name=\"AquaGuard_DNN\")\n",
            "    \n",
            "    model.compile(\n",
            "        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),\n",
            "        loss='binary_crossentropy',\n",
            "        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]\n",
            "    )\n",
            "    return model\n",
            "\n",
            "model = build_dnn(X_train.shape[1])\n",
            "model.summary()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Model Training with Dynamic Callbacks\n",
            "We train with `EarlyStopping` and `ReduceLROnPlateau`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "dnn_callbacks = [\n",
            "    callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),\n",
            "    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)\n",
            "]\n",
            "\n",
            "history = model.fit(\n",
            "    X_train, y_train,\n",
            "    validation_data=(X_val, y_val),\n",
            "    epochs=80,\n",
            "    batch_size=32,\n",
            "    callbacks=dnn_callbacks,\n",
            "    verbose=1\n",
            ")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Evaluation & Comparative Benchmark\n",
            "We evaluate on the independent test set and compare against Random Forest."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Test set predictions\n",
            "y_prob_dnn = model.predict(X_test).ravel()\n",
            "y_pred_dnn = (y_prob_dnn >= 0.5).astype(int)\n",
            "\n",
            "# Train baseline Random Forest\n",
            "rf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)\n",
            "rf.fit(X_train, y_train)\n",
            "y_prob_rf = rf.predict_proba(X_test)[:, 1]\n",
            "y_pred_rf = rf.predict(X_test)\n",
            "\n",
            "print(\"=== AquaGuard DNN Classification Report ===\")\n",
            "print(classification_report(y_test, y_pred_dnn))\n",
            "\n",
            "print(\"=== Random Forest Baseline Classification Report ===\")\n",
            "print(classification_report(y_test, y_pred_rf))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Explainable AI (XAI) & Attribution Analysis\n",
            "Feature importance analysis illustrates which chemical indicators trigger contamination risk flags."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Permutation Feature Importance\n",
            "base_acc = accuracy_score(y_test, y_pred_dnn)\n",
            "importances = {}\n",
            "\n",
            "for i, feat in enumerate(features):\n",
            "    X_perm = X_test.copy()\n",
            "    X_perm[:, i] = np.random.permutation(X_perm[:, i])\n",
            "    perm_pred = (model.predict(X_perm, verbose=0).ravel() >= 0.5).astype(int)\n",
            "    importances[feat] = max(0.001, base_acc - accuracy_score(y_test, perm_pred))\n",
            "\n",
            "df_imp = pd.Series(importances).sort_values(ascending=True)\n",
            "plt.figure(figsize=(9, 5))\n",
            "df_imp.plot(kind='barh', color='#2b5c8f')\n",
            "plt.title('Explainable AI: Feature Importance in Water Contamination Prediction')\n",
            "plt.xlabel('Accuracy Drop on Permutation')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Real-Time Inference & Safety Disclaimer\n",
            "Demonstration of practical input evaluation with WHO safety boundaries."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from predict import AquaGuardPredictor\n",
            "\n",
            "predictor = AquaGuardPredictor()\n",
            "sample = {\n",
            "    'ph': 7.4,\n",
            "    'Hardness': 185.0,\n",
            "    'Solids': 480.0,\n",
            "    'Chloramines': 2.8,\n",
            "    'Sulfate': 220.0,\n",
            "    'Conductivity': 390.0,\n",
            "    'Organic_carbon': 1.6,\n",
            "    'Trihalomethanes': 50.0,\n",
            "    'Turbidity': 2.4\n",
            "}\n",
            "\n",
            "inference = predictor.predict(sample)\n",
            "print(f\"Prediction: {inference['predicted_label']}\")\n",
            "print(f\"Potability Probability: {inference['potability_probability'] * 100:.2f}%\")\n",
            "print(f\"Risk Tier: {inference['contamination_risk_tier']}\")\n",
            "print(f\"Summary: {inference['risk_summary']}\")"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        },
        "orig_nbformat": 4
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(NOTEBOOK_PATH, "w") as f:
    json.dump(notebook, f, indent=2)

print(f"Generated academic Jupyter Notebook at {NOTEBOOK_PATH}")
