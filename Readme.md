# Predictive Sales Pipeline

A comprehensive machine learning and deep learning system for predicting enterprise sales opportunities with interpretability, ablation studies, and business-focused error analysis.

## Overview

This project implements **Phase 3: Hybrid Innovation**, combining:

- **Machine Learning Component**: Logistic Regression with feature preprocessing
- **Deep Learning Component**: Dual-path neural network (DualPathMLP) with separate numeric and categorical pathways
- **Hybrid Fusion**: 40% ML + 60% DL weighted probability ensemble

All analysis is **fully reproducible** with a single command and produces publication-ready visualizations and metrics.

## Project Structure

## How to Install

```text
predictive-sales project/
├── data/
│   ├── raw/                          # Original data sources
│   │   ├── accounts.csv
│   │   ├── products.csv
│   │   ├── sales_pipeline.csv
│   │   ├── sales_teams.csv
│   │   └── data_dictionary.csv
│   └── processed/
│       └── processed_sales_data.csv  # Training dataset (1,343 rows)
├── notebook/                         # Jupyter notebooks for EDA and analysis
│   ├── 01_EDA.ipynb
│   ├── 02_Modeling.ipynb
│   └── 03_Results_Analysis.ipynb
├── reports/                          # Documentation
│   ├── DL-Report.tex
│   └── Viva_Master_Guide.md
├── results/                          # Generated outputs (models, charts, metrics)
├── src/
│   ├── __init__.py
│   ├── process_data.py              # Data preprocessing utilities
│   └── phase3_pipeline.py           # Phase 3 implementation (900+ lines)
├── main.py                           # Single-command entry point
├── Requirements.txt                  # Python dependencies
├── .gitignore
└── Readme.md
```

## Installation

### Prerequisites

- Python 3.12+ (for optimal wheel availability)
- macOS, Linux, or WSL

### Setup Instructions

**1. Create and activate virtual environment:**

```bash
cd "/Users/amankumar/Desktop/predictive-sales project"
python3.12 -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies:**

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r Requirements.txt
```

This installs:

- Core: numpy (1.26.4), pandas (2.2.2), scikit-learn (1.4.2)
- Visualization: matplotlib (3.8.4), seaborn (0.13.2)
- Explainability: shap (0.44.1)
- Deep Learning: torch (2.3.1)
- Notebooks: jupyter (1.0.0), notebook (7.0.6)

**Execute the complete Phase 3 pipeline:**

```bash
cd "/Users/amankumar/Desktop/predictive-sales project"
source .venv/bin/activate
python main.py
```

**Runtime**: ~9 minutes on M-series MacBook Pro. All outputs are saved to `results/` directory.

## Expected Outputs

### Phase 3: Hybrid Innovation Analysis

#### 1. Model Comparison

- **`hybrid_model_comparison.csv`** - Metrics for ML-only, DL-only, and Hybrid models
  - Columns: Setup, F1, Precision, Recall, ROC-AUC, False Positives, False Negatives, Threshold
- **`hybrid_vs_ml_vs_dl_grouped_metrics.png`** - Grouped bar chart comparing all three approaches

#### 2. Ablation Study

- **`ablation_study_metrics.csv`** - Component contribution breakdown
- **`ablation_study_grouped_bar_chart.png`** - Visual comparison
- **Finding**: Removing DL reduces recall by 59.2%, removing ML increases false positives by 4.3% → both necessary

#### 3. Threshold Optimization

- **`threshold_analysis.csv`** - Performance metrics for thresholds 0.3-0.7
- **`threshold_analysis_hybrid.png`** - F1/Precision/Recall curves with optimal threshold highlighted
- **Optimal**: 0.3 (F1: 0.774, Recall: 1.0)

#### 4. SHAP Explainability

- **`shap_summary_top10_ml.png`** - Feature importance visualization (10 most impactful features)
- **`shap_top10_features_ml.csv`** - Feature rankings with SHAP values
- **`shap_business_interpretation.txt`** - Business-readable insights for stakeholders

#### 5. Error Segmentation Analysis

- **`error_segmentation_by_category.png`** - Heatmaps of error rates by:
  - Sector (10 categories)
  - Product Type (7 categories)
  - Price Range (4 quartiles)
- **`error_segmentation_summary.csv`** - Error rates per segment
- **Key Finding**: Model struggles most with mid-price deals ($1,096-$3,393) at 46.8% error rate

#### 6. Architecture Diagram

- **`architecture_diagram.png`** - Data flow visualization showing:
  - Numeric features pathway (8 features) → StandardScaler → 16-unit ReLU layer
  - Categorical features pathway (6 features) → OneHotEncoder → 32-unit ReLU layer
  - Fusion layer combining both paths → 64-unit ReLU → 32-unit ReLU → sigmoid output
  - Hybrid probability: 0.4×ML_prob + 0.6×DL_prob

## Model Architecture

### DualPathMLP (Custom Neural Network)

```
Numeric Path                 Categorical Path
├─ 8 features               ├─ 6 features
├─ StandardScaler           ├─ OneHotEncoder
├─ Dense(16, ReLU)          ├─ Dense(32, ReLU)
│                           │
└──────────────┬────────────┘
							 │
				 Dense(64, ReLU)
				 Dense(32, ReLU)
				 Dense(1, Sigmoid)
				 └─ DL Probability
```

### Hybrid Fusion

```
Hybrid Probability = 0.4 × LogisticRegression_prob + 0.6 × DualPathMLP_prob
```

## Key Metrics & Insights

### Overall Performance (Default Threshold: 0.5)

| Model      | F1        | Precision | Recall | ROC-AUC |
| ---------- | --------- | --------- | ------ | ------- |
| **Hybrid** | **0.745** | 0.628     | 0.916  | 0.520   |
| DL-only    | 0.766     | 0.632     | 0.973  | 0.521   |
| ML-only    | 0.473     | 0.643     | 0.374  | 0.514   |

### Top 5 Predictive Features (by SHAP Impact)

1. **log_employees** (0.095) - Decreases score at higher values
2. **engage_year** (0.080) - Timing of engagement matters
3. **year_established** (0.075) - Company maturity influence
4. **log_revenue** (0.057) - Increases score at higher values
5. **office_location_Panama** (0.024) - Geographic advantage

### Best & Worst Performing Segments

| Category | Best Segment | Error Rate | Worst Segment     | Error Rate |
| -------- | ------------ | ---------- | ----------------- | ---------- |
| Sector   | Marketing    | 29.3%      | Services          | 45.6%      |
| Product  | GTK 500      | 0%\*       | MG Advanced       | 46.8%      |
| Price    | Low ($0-550) | 37.1%      | Mid ($1.1K-$3.4K) | 46.8%      |

\*GTK 500: n=3 samples (limited data)

## Features Used

**Numeric Features (8)**:

- year_established, revenue, employees, sales_price
- log_revenue, log_employees, engage_year, engage_month

**Categorical Features (6)**:

- sector, product, manager, office_location, regional_office, series

## Development & Reproducibility

- **Random State**: Fixed at 42 across all operations
- **Data Split**: Stratified 80/20 train/test split
- **Preprocessing**: Fitted only on training data (prevents leakage)
- **Validation**: Early stopping with patience=20 on DL model
- **Environment**: Python 3.12 venv with pinned package versions

## Requirements

See `Requirements.txt` for complete dependency list. Key packages:

- numpy, pandas, scikit-learn, scipy
- matplotlib, seaborn
- shap (SHAP explainability)
- torch (deep learning framework)
- jupyter, notebook

## License

Proprietary - Predictive Sales Project
