# AquaGuard AI: Water Quality Dataset Documentation

## Overview
The dataset utilized in this project is based on the benchmark **Water Potability Dataset** (widely hosted on Kaggle and derived from environmental monitoring sources), enriched with contamination risk profiling aligned with World Health Organization (WHO) and United States Environmental Protection Agency (EPA) drinking water standards.

## Dataset Specifications
- **Total Records:** 3,276 water samples
- **Feature Count:** 9 continuous physicochemical parameters
- **Target Variables:**
  - `Potability`: Binary classification (0: Non-Potable / Contaminated, 1: Potable / Safe)
  - `Risk_Level`: Derived multi-class risk severity tier (0: Low Risk, 1: Moderate Risk, 2: High Contamination Risk)

## Feature Dictionary

| Feature | Units | WHO / EPA Safe Guidelines | Description |
| :--- | :--- | :--- | :--- |
| **ph** | Dimensionless (0-14) | 6.5 – 8.5 | Acid-base equilibrium. Acidic or highly alkaline water causes corrosion and gastrointestinal irritation. |
| **Hardness** | mg/L (CaCO₃ equivalent) | < 200 mg/L (desirable) | Capacity of water to precipitate soap; reflects calcium and magnesium ions. |
| **Solids (TDS)** | ppm (mg/L) | < 500 ppm (acceptable < 1000) | Total dissolved solids, mineral salts, and organic compounds. |
| **Chloramines** | ppm | < 4.0 ppm | Disinfection agent formed from ammonia and chlorine; elevated levels cause respiratory/skin irritation. |
| **Sulfate** | mg/L | < 250 mg/L | Naturally occurring sulfur minerals; excessive amounts induce laxative effects and bitterness. |
| **Conductivity** | μS/cm | < 400 μS/cm | Electrical conductivity correlated directly with ion concentrations and salinity. |
| **Organic_carbon** | ppm (mg/L) | < 2.0 ppm (WHO guideline) | Total Organic Carbon (TOC); proxy for decaying natural organic matter and synthetic chemical pollutants. |
| **Trihalomethanes** | μg/L | < 80 μg/L | Carcinogenic chemical by-products formed when chlorine reacts with organic compounds. |
| **Turbidity** | NTU (Nephelometric Turbidity Units) | < 5.0 NTU | Measure of water clarity and suspended particulate matter; can harbor pathogenic microorganisms. |

## Target Variable Definitions & Academic Integrity
- In standard academic datasets, `Potability` is a binary label (0 = non-potable, 1 = potable).
- **Academic Caution:** Real-world potability is not solely determined by chemical parameters; microbiological pathogens (e.g., *E. coli*, *Giardia*) require microbiological testing. AquaGuard AI models physicochemical risk and flags contamination indicators, providing transparent probability scores rather than declaring uncertified potability.

## Missing Value Strategy
Missing values in real-world environmental sampling occur due to sensor offline periods or reagent shortages. Preprocessing uses median imputation grouped by class label (or iterative median imputation fitted strictly on the training partition) to prevent data leakage.
