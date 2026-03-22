# Predictive Sales Pipeline & Failure Analysis

<div align="center">
  <h3><strong>Transforming Sales Probability into Predictable Revenue</strong></h3>
</div>

## 📖 Overview
The **Predictive Sales** project leverages advanced machine learning to predict the outcome of B2B sales opportunities (Won vs. Lost). Going beyond mere prediction, the core focus of this repository is a **rigorous failure analysis** to isolate where the model—and potentially the sales process itself—falls short, unearthing deeper insights to drive business strategy and resource allocation.

## ✨ Key Features
- **Comprehensive ETL Pipeline:** Intelligent merging and feature engineering across accounts, products, teams, and pipeline datasets.
- **Robust Modeling:** Gradient Boosting Classifier wrapped within a Scikit-Learn Pipeline for seamless preprocessing scaling and encoding.
- **In-Depth Failure Analysis:** Strategic examination of high-confidence False Positives (overestimated "Bad Bets") and False Negatives (undervalued "Missed Opportunities").
- **Feature Importance Tracking:** Mathematically identifies the principal factors driving successful conversions vs. lost deals.

## 📂 Project Structure
```text
predictive-sales project/
├── data/
│   ├── raw/             # Original CSV data files
│   └── processed/       # Engineered and merged data ready for modeling
├── notebook/
│   ├── 01_EDA.ipynb                 # Exploratory Data Analysis & visual insights
│   ├── 02_Modeling.ipynb            # Machine learning model & failure isolation
│   └── data_loading_and_cleaning.ipynb # Initial data sanitization
├── src/
│   ├── process_data.py              # Automated ETL script
│   └── create_modeling_notebook.py  # Notebook generation utility
├── reports/                         # LaTeX technical reports & presentation scripts
├── Requirements.txt                 # Dependency list
└── Readme.md                        # Project documentation
```

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/AmanKumar1411/AML_DL--Project.git
cd "predictive-sales project"
python3 -m venv venv
source venv/bin/activate
pip install -r Requirements.txt
```

### 2. Data Processing
Compile the raw CSV files into the unified dataset for modeling:
```bash
python src/process_data.py
```

### 3. Analytics & Modeling
Launch Jupyter Notebook to explore the analytical code:
```bash
jupyter notebook
```
- Open `notebook/01_EDA.ipynb` to review the data distributions and statistical findings.
- Open `notebook/02_Modeling.ipynb` to run the Gradient Boosting pipeline, observe classification performance, and dive into the critical failure analysis.

## 📊 Results & Analysis
The deployed Gradient Boosting model demonstrates highly capable performance in identifying `deal_stage` outcomes (ROC-AUC optimizations). Furthermore, the Failure Analysis isolates structural edge-cases where the sales team is disproportionately losing deals they are "expected" to win. This insight directly aids leadership in re-allocating resources away from low-probability efforts while aggressively pursuing systematically missed opportunities.

## 👨‍💻 Author
**Aman Kumar**
