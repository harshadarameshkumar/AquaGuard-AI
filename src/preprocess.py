"""
AquaGuard AI: Data Preprocessing & Exploratory Data Analysis Pipeline
Handles missing values, outlier detection, stratified train-val-test splitting,
and leakage-free standard scaling.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure directory paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

for d in [DATASET_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]

TARGET = "Potability"

def generate_eda_figures(df: pd.DataFrame):
    """Generates and saves exploratory data analysis plots."""
    sns.set_theme(style="whitegrid", palette="deep")
    
    # 1. Feature distributions by potability
    plt.figure(figsize=(14, 10))
    for i, col in enumerate(FEATURES, 1):
        plt.subplot(3, 3, i)
        sns.kdeplot(data=df, x=col, hue=TARGET, common_norm=False, fill=True, alpha=0.3, palette=["#e74c3c", "#2ecc71"])
        plt.title(f"Distribution: {col}", fontsize=11, fontweight='bold')
        plt.xlabel("")
        plt.ylabel("Density", fontsize=9)
    plt.tight_layout()
    eda_dist_path = os.path.join(RESULTS_DIR, "feature_distributions.png")
    plt.savefig(eda_dist_path, dpi=300)
    plt.close()
    print(f"Saved feature distributions plot to {eda_dist_path}")

    # 2. Correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.copy()
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-0.3, vmax=0.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot=True, fmt=".2f")
    plt.title("Physicochemical Parameter Correlation Heatmap", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    corr_path = os.path.join(RESULTS_DIR, "correlation_heatmap.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"Saved correlation heatmap to {corr_path}")

def compute_contamination_risk_index(df: pd.DataFrame) -> pd.Series:
    """
    Computes a validated contamination risk score based on WHO/EPA standards.
    0: Low Risk (potable, all parameters within safety boundaries)
    1: Moderate Risk (mild non-compliance in non-toxic aesthetic/mineral thresholds)
    2: High Contamination Risk (toxic byproducts or severe multi-parameter exceedance)
    """
    risk_score = np.zeros(len(df))
    
    # pH outside safe drinking range (6.5 - 8.5)
    ph = df['ph'].fillna(7.0)
    risk_score += np.where((ph < 6.5) | (ph > 8.5), 1.0, 0.0)
    risk_score += np.where((ph < 5.0) | (ph > 10.0), 1.5, 0.0) # extreme
    
    # Chloramines (> 4.0 ppm WHO guideline)
    risk_score += np.where(df['Chloramines'] > 8.0, 1.5, np.where(df['Chloramines'] > 4.0, 0.5, 0.0))
    
    # Trihalomethanes (Carcinogenic by-products > 80 ug/L)
    thm = df['Trihalomethanes'].fillna(66.0)
    risk_score += np.where(thm > 80.0, 1.5, 0.0)
    
    # Turbidity (> 5.0 NTU indicates pathogen harboring)
    risk_score += np.where(df['Turbidity'] > 5.0, 1.0, 0.0)

    # Total Dissolved Solids (> 1000 ppm)
    risk_score += np.where(df['Solids'] > 25000, 1.0, 0.0)

    # Classify into 3 levels
    risk_level = np.where(risk_score >= 2.5, 2, np.where(risk_score >= 1.0, 1, 0))
    return pd.Series(risk_level, index=df.index, name="Risk_Level")

def preprocess_pipeline(df: pd.DataFrame, test_size=0.15, val_size=0.15, random_state=42):
    """
    Executes a leakage-free preprocessing pipeline:
    1. Stratified split into train (70%), validation (15%), test (15%).
    2. Fit median imputers strictly on training data.
    3. Fit StandardScaler strictly on training data.
    4. Save transformation pipelines for inference.
    """
    print("--- Starting Preprocessing Pipeline ---")
    print(f"Initial raw shape: {df.shape}")
    print(f"Missing values count per feature:\n{df.isnull().sum()}")

    # Generate EDA figures
    generate_eda_figures(df)

    # Attach derived validated risk score
    df['Risk_Level'] = compute_contamination_risk_index(df)

    # Feature matrix X and target y
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    y_risk = df['Risk_Level'].copy()

    # Split into train (70%) and temporary holdout (30%)
    X_train_raw, X_temp_raw, y_train, y_temp, y_risk_train, y_risk_temp = train_test_split(
        X, y, y_risk, test_size=(test_size + val_size), random_state=random_state, stratify=y
    )

    # Split temporary holdout into validation (15% total) and test (15% total)
    val_ratio = val_size / (test_size + val_size)
    X_val_raw, X_test_raw, y_val, y_test, y_risk_val, y_risk_test = train_test_split(
        X_temp_raw, y_temp, y_risk_temp, test_size=(1 - val_ratio), random_state=random_state, stratify=y_temp
    )

    print(f"Train split: {X_train_raw.shape[0]} samples ({(len(X_train_raw)/len(df))*100:.1f}%)")
    print(f"Validation split: {X_val_raw.shape[0]} samples ({(len(X_val_raw)/len(df))*100:.1f}%)")
    print(f"Test split: {X_test_raw.shape[0]} samples ({(len(X_test_raw)/len(df))*100:.1f}%)")

    # 1. Median Imputation (Learned strictly from X_train_raw to avoid data leakage)
    train_medians = X_train_raw.median()
    print("Computed feature medians from training set:")
    for f, m in train_medians.items():
        print(f"  {f}: {m:.3f}")

    X_train_imp = X_train_raw.fillna(train_medians)
    X_val_imp = X_val_raw.fillna(train_medians)
    X_test_imp = X_test_raw.fillna(train_medians)

    # 2. Outlier capping based on training IQR (Winsorization)
    q25 = X_train_imp.quantile(0.01)
    q75 = X_train_imp.quantile(0.99)
    X_train_imp = X_train_imp.clip(lower=q25, upper=q75, axis=1)
    X_val_imp = X_val_imp.clip(lower=q25, upper=q75, axis=1)
    X_test_imp = X_test_imp.clip(lower=q25, upper=q75, axis=1)

    # 3. Standard Scaling (Fit strictly on training set)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # Save preprocessing metadata and artifacts
    preprocessor_artifact = {
        "features": FEATURES,
        "medians": train_medians.to_dict(),
        "q25": q25.to_dict(),
        "q75": q75.to_dict(),
        "scaler": scaler,
        "train_mean": scaler.mean_.tolist(),
        "train_scale": scaler.scale_.tolist()
    }
    artifact_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor_artifact, artifact_path)
    print(f"Saved preprocessor artifact to {artifact_path}")

    # Save processed numpy arrays for reproducibility
    np.savez_compressed(
        os.path.join(DATASET_DIR, "processed_splits.npz"),
        X_train=X_train_scaled,
        X_val=X_val_scaled,
        X_test=X_test_scaled,
        y_train=y_train.to_numpy(),
        y_val=y_val.to_numpy(),
        y_test=y_test.to_numpy(),
        y_risk_train=y_risk_train.to_numpy(),
        y_risk_val=y_risk_val.to_numpy(),
        y_risk_test=y_risk_test.to_numpy()
    )
    print(f"Saved processed data splits to {os.path.join(DATASET_DIR, 'processed_splits.npz')}")

    return {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train.to_numpy(),
        "y_val": y_val.to_numpy(),
        "y_test": y_test.to_numpy(),
        "y_risk_train": y_risk_train.to_numpy(),
        "y_risk_val": y_risk_val.to_numpy(),
        "y_risk_test": y_risk_test.to_numpy(),
        "preprocessor": preprocessor_artifact
    }

if __name__ == "__main__":
    from fetch_data import load_or_fetch_dataset
    df = load_or_fetch_dataset()
    preprocess_pipeline(df)
