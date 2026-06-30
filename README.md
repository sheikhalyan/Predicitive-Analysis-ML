# ⚙️ Predictive Maintenance Classification

### Multi-target machine learning system predicting equipment failure and failure type from sensor data

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)

> A comparative machine learning study predicting both **whether** industrial equipment will fail and **what type** of failure will occur, using sensor telemetry data. Six classification algorithms are trained and benchmarked head-to-head on the same multi-target prediction task.

---

## Problem Statement

Industrial equipment failure is costly and often preventable. Using real-time sensor readings (temperature, rotational speed, torque, tool wear), this project predicts:

1. **Target** — will the machine fail or not (binary classification)
2. **Failure Type** — if it fails, which of 5 failure modes is occurring (multi-class classification)

Both targets are predicted **simultaneously** using `MultiOutputClassifier`, rather than as two separate models — a more realistic setup for production predictive maintenance systems.

---

## Dataset

[**Machine Predictive Maintenance Classification**](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) — Kaggle

- **10,000 data points**, 14 original features
- **Sensor features:** Air temperature, Process temperature, Rotational speed, Torque, Tool wear
- **Product metadata:** Product ID, Quality variant (Low / Medium / High)
- **Targets:** Binary failure flag + 5-class failure type (No Failure, Power Failure, Tool Wear Failure, Overstrain Failure, Random Failures, Heat Dissipation Failure)
- **Class imbalance:** ~96.6% no-failure vs ~3.4% failure cases — a realistic and challenging imbalance for predictive maintenance

---

## Models Compared

Six classifiers were trained and evaluated on the identical train/test split (70/30) to identify the best-performing approach for this multi-target task:

| Model | Accuracy — Target | Accuracy — Failure Type |
|-------|--------------------|--------------------------|
| **Gradient Boosting Machine (GBM)** | **98.53%** | **98.13%** |
| **Random Forest** | 98.50% | 98.20% |
| Decision Tree | 98.03% | 97.47% |
| Logistic Regression | 97.40% | 96.77% |
| K-Nearest Neighbors (KNN) | 97.13% | 96.93% |
| Support Vector Classifier (SVC) | 96.90% | 96.77% |

**Winner: Gradient Boosting Machine**, narrowly outperforming Random Forest on the primary failure-detection target, with Random Forest slightly ahead on failure-type classification — both well-suited to this kind of complex, imbalanced sensor data.

---

## Feature Importance

Using Random Forest's built-in feature importance scoring, the most predictive signals for failure were:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Torque [Nm] | 0.258 |
| 2 | Rotational speed [rpm] | 0.227 |
| 3 | UDI (unique identifier) | 0.129 |
| 4 | Tool wear [min] | 0.122 |
| 5 | Product ID | 0.097 |
| 6 | Air temperature [K] | 0.080 |
| 7 | Process temperature [K] | 0.073 |
| 8 | Type (quality variant) | 0.014 |

**Torque and rotational speed dominate** as the strongest predictors of equipment failure — intuitive given they directly reflect mechanical stress on the equipment.

---

## Methodology

```
Raw sensor data (10,000 rows)
        ↓
Data cleaning & validation (null checks, type checks)
        ↓
Label encoding (categorical → numeric: Type, Failure Type)
        ↓
Exploratory Data Analysis (distribution plots, class balance)
        ↓
Train/Test Split (70% / 30%)
        ↓
Train 6 classifiers via MultiOutputClassifier
        ↓
Evaluate accuracy per target (Target + Failure Type)
        ↓
Feature importance analysis (Random Forest)
        ↓
Compare & select best-performing model
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Language** | Python |
| **ML Framework** | scikit-learn |
| **Data Manipulation** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook |

**Algorithms implemented:** Gradient Boosting Classifier, Decision Tree, Random Forest, Support Vector Machine (SVC), Logistic Regression, K-Nearest Neighbors

---

## Project Structure

```
predictive-maintenance-ml/
│
├── Predictive_Analysis.ipynb                              # Main analysis notebook
├── predictive_maintenance.csv                              # Dataset (10,000 rows)
├── Predictive Maintenance Proposal.pdf                     # Project proposal
├── Predictive Maintenance Classification using ML.pptx     # Presentation deck
│
└── Visualizations/
    ├── target_variable_distribution.png                    # Class balance (Target)
    ├── failure_type_distribution.png                       # Class balance (Failure Type)
    ├── feature_importance.png                               # Feature importance bar chart
    ├── Air temperature [K]_distribution.png
    ├── Process temperature [K]_distribution.png
    ├── Rotational speed [rpm]_distribution.png
    └── Torque [Nm]_distribution.png
```

---

## Getting Started

### Prerequisites
- Python 3.x
- Jupyter Notebook or JupyterLab

### Setup

1. Clone the repository
   ```bash
   git clone https://github.com/sheikhalyan/Predicitive-Analysis-ML.git
   cd Predicitive-Analysis-ML
   ```

2. Install dependencies
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. Launch the notebook
   ```bash
   jupyter notebook Predictive_Analysis.ipynb
   ```

4. Run all cells to reproduce the full analysis, from EDA through model comparison

---

## Key Takeaways

This project demonstrates an end-to-end ML workflow — from raw sensor data to a benchmarked, production-considerable model — with particular attention to:

- **Multi-target prediction** rather than treating failure detection and failure classification as separate problems
- **Class imbalance awareness** — with only ~3.4% positive failure cases, accuracy alone is reported alongside the understanding that imbalanced classification requires careful interpretation
- **Model comparison rigor** — six algorithms evaluated under identical conditions rather than cherry-picking one approach
- **Feature importance interpretation** — translating model internals into actionable engineering insight (torque and RPM as primary failure drivers)

---

## Author

**Sheikh Alyan** — BS Computer Science, PAF-KIET

[![GitHub](https://img.shields.io/badge/GitHub-@sheikhalyan-181717?style=flat-square&logo=github)](https://github.com/sheikhalyan)
