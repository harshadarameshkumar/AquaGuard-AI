"""
AquaGuard AI: Deep Neural Network (DNN) Training Module
Trains a Deep Feedforward Neural Network using TensorFlow / Keras with:
- Dense layers (128 -> 64 -> 32)
- Batch Normalization & Dropout regularization
- Adam optimizer with ReduceLROnPlateau & EarlyStopping
- ModelCheckpointing
- Baseline comparative models: Random Forest & Gradient Boosting
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

for d in [DATASET_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

def build_dnn_architecture_1(input_dim: int):
    """
    Standard Architecture (Config 1):
    Input -> Dense(128, ReLU) -> BatchNorm -> Dropout(0.3) ->
    Dense(64, ReLU) -> Dropout(0.2) -> Dense(32, ReLU) -> Dense(1, Sigmoid)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ], name="AquaGuard_DNN_Config1")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
    )
    return model

def build_dnn_architecture_2(input_dim: int):
    """
    Deeper / Wider Architecture (Config 2 for experimental comparison):
    Dense(256) -> BatchNorm -> Dropout(0.4) ->
    Dense(128) -> BatchNorm -> Dropout(0.3) ->
    Dense(64) -> Dropout(0.2) -> Dense(16) -> Dense(1, Sigmoid)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(2e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(2e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ], name="AquaGuard_DNN_Config2_Deep")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
    )
    return model

def plot_training_curves(history, output_prefix=""):
    """Plots and saves loss and accuracy curves."""
    epochs = range(1, len(history.history['loss']) + 1)
    
    # 1. Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history.history['loss'], 'b-', lw=2, label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r--', lw=2, label='Validation Loss')
    plt.title('AquaGuard DNN Loss Progression Across Epochs', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('Binary Cross-Entropy Loss', fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(RESULTS_DIR, f"{output_prefix}loss.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"Saved loss curve to {loss_path}")

    # 2. Accuracy Curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history.history['accuracy'], 'g-', lw=2, label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'm--', lw=2, label='Validation Accuracy')
    plt.title('AquaGuard DNN Classification Accuracy Progression', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_path = os.path.join(RESULTS_DIR, f"{output_prefix}accuracy.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"Saved accuracy curve to {acc_path}")

def train_pipeline():
    # Load processed splits
    splits_path = os.path.join(DATASET_DIR, "processed_splits.npz")
    if not os.path.exists(splits_path):
        from fetch_data import load_or_fetch_dataset
        from preprocess import preprocess_pipeline
        df = load_or_fetch_dataset()
        preprocess_pipeline(df)
    
    data = np.load(splits_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    input_dim = X_train.shape[1]
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # Check for TensorFlow
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        print("TensorFlow successfully detected. Version:", tf.__version__)
        has_tf = True
    except ImportError:
        print("TensorFlow not installed. Using Scikit-Learn Deep MLP as fallback.")
        has_tf = False

    metrics_summary = {}

    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = {cls: float(w) for cls, w in zip(classes, weights)}
    print(f"Computed Balanced Class Weights: {class_weight_dict}")

    if has_tf:
        # Train Config 1 (Standard Suggested Architecture)
        print("\n--- Training Deep Neural Network (Architecture 1: 128-64-32) ---")
        model1 = build_dnn_architecture_1(input_dim)
        model1.summary()

        checkpoint_path = os.path.join(MODELS_DIR, "aquaguard_dnn_best.keras")
        callbacks1 = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5, verbose=1),
            ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=0)
        ]

        history1 = model1.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            class_weight=class_weight_dict,
            callbacks=callbacks1,
            verbose=1
        )

        plot_training_curves(history1)
        model1.save(os.path.join(MODELS_DIR, "aquaguard_dnn_final.keras"))

        # Find best threshold on validation set
        val_prob1 = model1.predict(X_val).ravel()
        best_thresh = 0.5
        best_f1 = 0
        for t in np.linspace(0.3, 0.7, 41):
            f1_t = f1_score(y_val, (val_prob1 >= t).astype(int), zero_division=0)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_thresh = t
        print(f"Optimal decision threshold calibrated on validation split: {best_thresh:.3f} (Val F1: {best_f1:.4f})")

        # Evaluate Model 1 on test set
        test_pred_prob1 = model1.predict(X_test).ravel()
        test_pred1 = (test_pred_prob1 >= best_thresh).astype(int)

        metrics_summary["DNN_Config_1"] = {
            "accuracy": float(accuracy_score(y_test, test_pred1)),
            "precision": float(precision_score(y_test, test_pred1, zero_division=0)),
            "recall": float(recall_score(y_test, test_pred1, zero_division=0)),
            "f1_score": float(f1_score(y_test, test_pred1, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, test_pred_prob1)),
            "threshold": float(best_thresh)
        }

        # Train Config 2 (Deeper Architecture Comparison)
        print("\n--- Training Deep Neural Network (Architecture 2: 256-128-64-16) ---")
        model2 = build_dnn_architecture_2(input_dim)
        callbacks2 = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)
        ]
        history2 = model2.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=80,
            batch_size=32,
            class_weight=class_weight_dict,
            callbacks=callbacks2,
            verbose=0
        )
        test_pred_prob2 = model2.predict(X_test).ravel()
        test_pred2 = (test_pred_prob2 >= best_thresh).astype(int)
        metrics_summary["DNN_Config_2"] = {
            "accuracy": float(accuracy_score(y_test, test_pred2)),
            "precision": float(precision_score(y_test, test_pred2, zero_division=0)),
            "recall": float(recall_score(y_test, test_pred2, zero_division=0)),
            "f1_score": float(f1_score(y_test, test_pred2, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, test_pred_prob2)),
            "threshold": float(best_thresh)
        }

    else:
        # Sklearn MLP Classifier as robust neural fallback
        from sklearn.neural_network import MLPClassifier
        print("\n--- Training Scikit-learn MLP Classifier (Architecture: 128-64-32) ---")
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                            solver='adam', max_iter=200, random_state=42, early_stopping=True)
        mlp.fit(X_train, y_train)
        joblib.dump(mlp, os.path.join(MODELS_DIR, "aquaguard_dnn_model.joblib"))
        test_pred_prob = mlp.predict_proba(X_test)[:, 1]
        test_pred = mlp.predict(X_test)
        metrics_summary["DNN_Config_1"] = {
            "accuracy": float(accuracy_score(y_test, test_pred)),
            "precision": float(precision_score(y_test, test_pred, zero_division=0)),
            "recall": float(recall_score(y_test, test_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, test_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, test_pred_prob))
        }

    # Train Baseline 1: Random Forest Classifier
    print("\n--- Training Baseline Model: Random Forest ---")
    rf = RandomForestClassifier(n_estimators=150, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred_prob = rf.predict_proba(X_test)[:, 1]
    rf_pred = (rf_pred_prob >= 0.45).astype(int)
    joblib.dump(rf, os.path.join(MODELS_DIR, "baseline_rf.joblib"))

    metrics_summary["Random_Forest"] = {
        "accuracy": float(accuracy_score(y_test, rf_pred)),
        "precision": float(precision_score(y_test, rf_pred, zero_division=0)),
        "recall": float(recall_score(y_test, rf_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, rf_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, rf_pred_prob))
    }

    # Train Baseline 2: Gradient Boosting Classifier
    print("\n--- Training Baseline Model: Gradient Boosting ---")
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred_prob = gb.predict_proba(X_test)[:, 1]
    gb_pred = gb.predict(X_test)
    joblib.dump(gb, os.path.join(MODELS_DIR, "baseline_gb.joblib"))

    metrics_summary["Gradient_Boosting"] = {
        "accuracy": float(accuracy_score(y_test, gb_pred)),
        "precision": float(precision_score(y_test, gb_pred, zero_division=0)),
        "recall": float(recall_score(y_test, gb_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, gb_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, gb_pred_prob))
    }

    # Save metrics summary to file
    metrics_path = os.path.join(RESULTS_DIR, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Saved model metrics summary to {metrics_path}")
    print("\n=== Model Performance Comparison ===")
    for model_name, m in metrics_summary.items():
        print(f"{model_name:20s} | Acc: {m['accuracy']:.4f} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | F1: {m['f1_score']:.4f} | AUC: {m['roc_auc']:.4f}")

    return metrics_summary

if __name__ == "__main__":
    train_pipeline()
