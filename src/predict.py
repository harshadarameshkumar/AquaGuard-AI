"""
AquaGuard AI: Inference & Water Contamination Risk Prediction Engine
Accepts physicochemical parameters, validates physiological and physical limits,
executes neural inference, assesses multi-parameter contamination risk,
and delivers transparent, calibrated predictions with safety disclaimers.
"""

import os
import sys
import json
import numpy as np
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

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

# Physical and physiological bounds for input validation
VALIDATION_RANGES = {
    "ph": (0.0, 14.0, "Dimensionless pH unit"),
    "Hardness": (10.0, 600.0, "mg/L (as CaCO3)"),
    "Solids": (100.0, 70000.0, "ppm (Total Dissolved Solids)"),
    "Chloramines": (0.0, 20.0, "ppm (mg/L)"),
    "Sulfate": (50.0, 600.0, "mg/L"),
    "Conductivity": (50.0, 1000.0, "uS/cm"),
    "Organic_carbon": (1.0, 40.0, "ppm (mg/L Total Organic Carbon)"),
    "Trihalomethanes": (0.0, 200.0, "ug/L"),
    "Turbidity": (0.0, 15.0, "NTU (Nephelometric Turbidity Units)")
}

# Regulatory guidelines (WHO / US-EPA benchmark thresholds)
REGULATORY_STANDARDS = {
    "ph": {"min": 6.5, "max": 8.5, "unit": "pH", "desc": "WHO Aesthetic & Safety Range"},
    "Hardness": {"max": 300.0, "unit": "mg/L", "desc": "High hardness may cause scale/mineral deposits"},
    "Solids": {"max": 1000.0, "unit": "ppm", "desc": "WHO Palatability Threshold"},
    "Chloramines": {"max": 4.0, "unit": "ppm", "desc": "EPA Maximum Residual Disinfectant Level"},
    "Sulfate": {"max": 250.0, "unit": "mg/L", "desc": "WHO Secondary Drinking Water Limit"},
    "Conductivity": {"max": 400.0, "unit": "uS/cm", "desc": "WHO Standard Conductivity Guide"},
    "Organic_carbon": {"max": 2.0, "unit": "ppm", "desc": "WHO Natural Organic Matter Alert Threshold"},
    "Trihalomethanes": {"max": 80.0, "unit": "ug/L", "desc": "EPA Maximum Contaminant Level (Carcinogenic THMs)"},
    "Turbidity": {"max": 5.0, "unit": "NTU", "desc": "WHO Filtration & Microbial Protection Benchmark"}
}

class AquaGuardPredictor:
    def __init__(self):
        self.preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
        self.keras_model_path = os.path.join(MODELS_DIR, "aquaguard_dnn_best.keras")
        if not os.path.exists(self.keras_model_path):
            self.keras_model_path = os.path.join(MODELS_DIR, "aquaguard_dnn_final.keras")
        self.rf_path = os.path.join(MODELS_DIR, "baseline_rf.joblib")
        
        self.preprocessor = None
        self.dnn_model = None
        self.rf_model = None
        self._load_artifacts()

    def _load_artifacts(self):
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Preprocessor artifact missing at {self.preprocessor_path}. Run preprocess.py first.")
        self.preprocessor = joblib.load(self.preprocessor_path)

        if os.path.exists(self.keras_model_path):
            try:
                import tensorflow as tf
                self.dnn_model = tf.keras.models.load_model(self.keras_model_path)
            except Exception as e:
                print(f"Warning: Could not load Keras model: {e}")

        if self.dnn_model is None:
            joblib_dnn = os.path.join(MODELS_DIR, "aquaguard_dnn_model.joblib")
            if os.path.exists(joblib_dnn):
                self.dnn_model = joblib.load(joblib_dnn)

        if os.path.exists(self.rf_path):
            self.rf_model = joblib.load(self.rf_path)

    def validate_inputs(self, sample_dict: dict):
        """Validates all inputs against physical and valid ranges."""
        validated = {}
        violations = []

        for feature in FEATURES:
            if feature not in sample_dict or sample_dict[feature] is None:
                # Use training median fallback if missing
                imputed = self.preprocessor["medians"][feature]
                validated[feature] = float(imputed)
                violations.append(f"Notice: '{feature}' was not provided; imputed with population training median ({imputed:.2f}).")
                continue

            try:
                val = float(sample_dict[feature])
            except (ValueError, TypeError):
                raise ValueError(f"Invalid input for '{feature}': must be a numeric value, got '{sample_dict[feature]}'.")

            min_val, max_val, unit = VALIDATION_RANGES[feature]
            if val < min_val or val > max_val:
                raise ValueError(
                    f"Parameter '{feature}' value {val} is outside valid plausible instrument boundaries [{min_val}, {max_val}] {unit}."
                )

            validated[feature] = val

        return validated, violations

    def evaluate_regulatory_compliance(self, sample: dict):
        """Audits water parameters against WHO / EPA drinking water standards."""
        flags = []
        severity_points = 0.0

        ph = sample["ph"]
        if ph < 6.5 or ph > 8.5:
            severity = "Severe" if (ph < 5.0 or ph > 9.5) else "Moderate"
            flags.append({
                "parameter": "ph",
                "value": ph,
                "standard": "6.5 - 8.5",
                "status": f"Non-Compliant ({severity})",
                "impact": "Acidic or caustic corrosive potential; mucous membrane irritation."
            })
            severity_points += 1.5 if severity == "Severe" else 1.0

        chlor = sample["Chloramines"]
        if chlor > 4.0:
            severity = "Critical" if chlor > 8.0 else "Elevated"
            flags.append({
                "parameter": "Chloramines",
                "value": chlor,
                "standard": "<= 4.0 ppm",
                "status": f"Non-Compliant ({severity})",
                "impact": "Skin sensitization, respiratory distress, and chlorine toxicity."
            })
            severity_points += 1.5 if severity == "Critical" else 0.8

        thm = sample["Trihalomethanes"]
        if thm > 80.0:
            flags.append({
                "parameter": "Trihalomethanes",
                "value": thm,
                "standard": "<= 80.0 ug/L",
                "status": "Non-Compliant (Hazardous)",
                "impact": "Volatile organohalogen byproducts linked to chronic oncological risks."
            })
            severity_points += 1.5

        turb = sample["Turbidity"]
        if turb > 5.0:
            flags.append({
                "parameter": "Turbidity",
                "value": turb,
                "standard": "<= 5.0 NTU",
                "status": "Non-Compliant (Alert)",
                "impact": "Suspended colloidal matter shielding cryptosporidium or coliform bacteria."
            })
            severity_points += 1.0

        solids = sample["Solids"]
        if solids > 1000.0:
            flags.append({
                "parameter": "Solids (TDS)",
                "value": solids,
                "standard": "<= 1000 ppm",
                "status": "Elevated Mineralization",
                "impact": "High salinity, gastrointestinal distress, and scale accumulation."
            })
            severity_points += 0.5

        return flags, severity_points

    def predict(self, sample_dict: dict):
        """
        Executes full inference pipeline for single sample.
        Returns:
            - predicted_class: 0 (Non-Potable / Contaminated) or 1 (Potable / Safe)
            - probability: confidence of potability [0.0 - 1.0]
            - contamination_risk_tier: "LOW_RISK", "MODERATE_RISK", "HIGH_RISK"
            - regulatory_flags: list of non-compliant parameters
            - warnings: input notices
            - limitations: legal & academic disclaimer
        """
        validated_sample, warnings = self.validate_inputs(sample_dict)
        raw_values = np.array([validated_sample[f] for f in FEATURES]).reshape(1, -1)

        # Scale using training scaler
        scaled_values = self.preprocessor["scaler"].transform(raw_values)

        # Predict probability
        if self.dnn_model is not None:
            if hasattr(self.dnn_model, 'predict_proba'):
                prob = float(self.dnn_model.predict_proba(scaled_values)[0, 1])
            else:
                prob = float(self.dnn_model.predict(scaled_values, verbose=0)[0, 0])
        elif self.rf_model is not None:
            prob = float(self.rf_model.predict_proba(scaled_values)[0, 1])
        else:
            raise RuntimeError("No trained model weights found in models directory.")

        pred_class = int(prob >= 0.5)

        # Audit regulatory standards
        reg_flags, severity_score = self.evaluate_regulatory_compliance(validated_sample)

        # Determine Risk Category
        if severity_score >= 2.0 or prob < 0.35:
            risk_tier = "HIGH_CONTAMINATION_RISK"
            risk_summary = "Water sample exhibits significant contamination indicators or fails critical safety guidelines."
            action_recommendation = "Do not consume without advanced tertiary purification (reverse osmosis + UV disinfection)."
        elif severity_score >= 0.8 or prob < 0.55:
            risk_tier = "MODERATE_RISK"
            risk_summary = "Water sample indicates marginal compliance with certain non-optimal physicochemical markers."
            action_recommendation = "Filtration and chemical disinfection advised before consumption."
        else:
            risk_tier = "LOW_RISK"
            risk_summary = "Physicochemical parameters reside within acceptable drinking water safety envelopes."
            action_recommendation = "Potable based on physicochemical metrics. Ensure routine microbiological validation."

        result = {
            "predicted_class": pred_class,
            "predicted_label": "Potable (Safe Physicochemical Profile)" if pred_class == 1 else "Non-Potable (Elevated Risk / Contaminated)",
            "potability_probability": round(prob, 4),
            "contamination_risk_tier": risk_tier,
            "risk_summary": risk_summary,
            "action_recommendation": action_recommendation,
            "severity_score": round(severity_score, 2),
            "regulatory_violations": reg_flags,
            "input_warnings": warnings,
            "input_parameters": validated_sample,
            "academic_limitations": (
                "IMPORTANT LIMITATION & DISCLAIMER: This prediction is generated by an academic Deep Neural Network "
                "trained on physicochemical parameters (pH, TDS, Chloramines, Sulfates, THMs, Turbidity). "
                "It DOES NOT evaluate microbiological pathogens (e.g., Legionella, E. coli, Norovirus) or heavy metals (Lead, Arsenic, Mercury). "
                "Never present or rely on these computational inferences as a certified public health or drinking water safety assessment."
            )
        }
        return result

def cli_test():
    predictor = AquaGuardPredictor()
    
    # Example 1: Clean safe water sample
    safe_sample = {
        "ph": 7.2,
        "Hardness": 180.0,
        "Solids": 450.0,
        "Chloramines": 2.5,
        "Sulfate": 210.0,
        "Conductivity": 380.0,
        "Organic_carbon": 1.8,
        "Trihalomethanes": 45.0,
        "Turbidity": 2.1
    }
    print("=== Testing Sample 1: Potable Water ===")
    res1 = predictor.predict(safe_sample)
    print(json.dumps(res1, indent=2))

    # Example 2: Contaminated water sample
    dirty_sample = {
        "ph": 4.1,
        "Hardness": 380.0,
        "Solids": 32000.0,
        "Chloramines": 9.5,
        "Sulfate": 420.0,
        "Conductivity": 650.0,
        "Organic_carbon": 18.5,
        "Trihalomethanes": 125.0,
        "Turbidity": 8.4
    }
    print("\n=== Testing Sample 2: Contaminated Industrial/Polluted Water ===")
    res2 = predictor.predict(dirty_sample)
    print(json.dumps(res2, indent=2))

if __name__ == "__main__":
    cli_test()
