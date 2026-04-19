
import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# Predictive Modeling (AML + DL) & Failure Analysis

## Goal
Predict `deal_stage` (`target`: Won/Lost) with a clean, controlled pipeline that covers:
1. **Baseline ML**: Logistic Regression
2. **Advanced ML**: SVM with RBF kernel
3. **Deep Learning**: MLPClassifier (primary model)

## Scope Control
- Remove unnecessary ensembles and excessive model comparisons.
- Keep only essential models aligned with course objectives.
- Prioritize F1, ROC AUC, precision, recall, and failure analysis over accuracy."""

code_imports = """import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

os.makedirs("../results", exist_ok=True)

try:
    df = pd.read_csv("../data/processed/processed_sales_data.csv")
except FileNotFoundError:
    df = pd.read_csv("data/processed/processed_sales_data.csv")

print(f"Loaded data shape: {df.shape}")"""

text_pipeline = """## 1. Data + Preprocessing Pipeline (Retained)
This keeps the cleaned/merged data workflow and the established leakage-safe feature set:
- Retain engineered/log-transformed features (`log_revenue`, `log_employees`)
- Retain leakage removal (`close_*`, `deal_duration*`, `log_close_value` excluded)
- Retain preprocessing via `ColumnTransformer` with:
  - `StandardScaler` for numerics
  - `OneHotEncoder` for categoricals"""

code_pipeline = """features = [
    'sector', 'year_established', 'revenue', 'employees', 'office_location',
    'series', 'sales_price', 'product', 'manager', 'regional_office',
    'engage_year', 'engage_month', 'log_revenue', 'log_employees'
]
target = 'target'

df = df.dropna(subset=features + [target]).copy()
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

categorical_features = ['sector', 'office_location', 'series', 'product', 'manager', 'regional_office']
numeric_features = ['year_established', 'revenue', 'employees', 'sales_price', 'engage_year', 'engage_month', 'log_revenue', 'log_employees']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ]
)

print(f'Train shape: {X_train.shape} | Test shape: {X_test.shape}')
print('Target balance in train (normalized):')
print(y_train.value_counts(normalize=True).sort_index())"""

text_models = """## 2. Essential Models Only
- **Baseline**: Logistic Regression (`class_weight='balanced'`)
- **AML model**: SVM with RBF kernel (`class_weight='balanced'`, light tuning only)
- **Primary DL model**: MLPClassifier with hidden layers `(64, 32)`, ReLU hidden activations, and sigmoid output behavior for binary classification"""

code_models = """models = {
    'Logistic Regression (Baseline)': Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
        ]
    ),
    'SVM (RBF)': Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42,
            )),
        ]
    ),
    'MLP (Primary)': Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=15,
                random_state=42,
            )),
        ]
    ),
}

model_tags = {
    'Logistic Regression (Baseline)': 'baseline_ml_logistic_regression',
    'SVM (RBF)': 'aml_svm_rbf',
    'MLP (Primary)': 'dl_mlp_primary',
}

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    row = {
        'Model': name,
        'F1': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'ROC_AUC': roc_auc_score(y_test, y_prob),
    }
    results.append(row)
    trained_models[name] = model

    tag = model_tags[name]

    cm_model = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_model, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'../results/confusion_matrix_{tag}.png', bbox_inches='tight')
    plt.close()

    with open(f'../results/classification_report_{tag}.txt', 'w') as report_file:
        report_file.write(f'Model: {name}\\n')
        report_file.write(classification_report(y_test, y_pred, digits=4))

results_df = pd.DataFrame(results).sort_values('F1', ascending=False)
print('Model comparison (essential only):')
display(results_df)

with open('../results/model_metrics_aml_dl_all_models.txt', 'w') as f:
    f.write('Essential Models (AML + DL)\\n')
    f.write(results_df.to_string(index=False))

# Keep this file for backward compatibility with existing report references.
with open('../results/model_metrics.txt', 'w') as f:
    f.write('Essential Models (AML + DL)\\n')
    f.write(results_df.to_string(index=False))

primary_model_name = 'MLP (Primary)'
primary_model = trained_models[primary_model_name]
print(f'Primary model selected for detailed analysis: {primary_model_name}')"""

text_eval = """## 3. Primary Evaluation (F1-First)
Accuracy is intentionally de-emphasized. We focus on F1, ROC AUC, precision, and recall.

Optional threshold tuning is included to reduce false positives when needed."""

code_eval = """# Keep 0.50 for default behavior. Try 0.60 or 0.65 to reduce false positives.
MLP_THRESHOLD = 0.50

y_prob = primary_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= MLP_THRESHOLD).astype(int)

print(f'MLP decision threshold: {MLP_THRESHOLD:.2f}')
print(f'F1: {f1_score(y_test, y_pred):.4f}')
print(f'Precision: {precision_score(y_test, y_pred):.4f}')
print(f'Recall: {recall_score(y_test, y_pred):.4f}')
print(f'ROC AUC: {roc_auc_score(y_test, y_prob):.4f}')

print('Classification report for MLP (Primary):')
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - MLP (Primary)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
threshold_tag = str(MLP_THRESHOLD).replace('.', 'p')
plt.savefig(f'../results/confusion_matrix_dl_mlp_primary_threshold_{threshold_tag}.png', bbox_inches='tight')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f'TN={tn}, FP={fp}, FN={fn}, TP={tp}')"""

text_validation = """## 4. Validation Check (Optional CV)
Train/test split is retained. Optional cross-validation can be enabled for extra robustness while avoiding leakage by using full pipelines."""

code_validation = """RUN_CV = False

if RUN_CV:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(primary_model, X, y, cv=cv, scoring='f1', n_jobs=-1)
    print(f'CV F1 scores: {np.round(cv_scores, 4)}')
    print(f'CV F1 mean: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')
else:
    print('Cross-validation skipped (RUN_CV=False).')"""

text_failure = """## 5. Failure Analysis
- **False Positives**: bad bets (predicted Won, actually Lost)
- **False Negatives**: missed opportunities (predicted Lost, actually Won)

We analyze failures by sector, product, pricing bands, and high-confidence errors."""

code_failure = """failure_df = X_test.copy()
failure_df['Actual'] = y_test.values
failure_df['Predicted'] = y_pred
failure_df['Probability'] = y_prob

failure_df['Result'] = 'Correct'
failure_df.loc[(failure_df['Actual'] == 0) & (failure_df['Predicted'] == 1), 'Result'] = 'False Positive (Bad Bet)'
failure_df.loc[(failure_df['Actual'] == 1) & (failure_df['Predicted'] == 0), 'Result'] = 'False Negative (Missed Opp)'

error_df = failure_df[failure_df['Result'] != 'Correct'].copy()
print(f'Total errors: {len(error_df)} / {len(failure_df)}')

if len(error_df) > 0:
    print('Error distribution by sector:')
    print(pd.crosstab(error_df['sector'], error_df['Result']).sort_index())

    print('Error distribution by product:')
    print(pd.crosstab(error_df['product'], error_df['Result']).sort_index())

    failure_df['price_band'] = pd.qcut(failure_df['sales_price'], q=4, duplicates='drop')
    print('Error distribution by pricing band:')
    print(pd.crosstab(failure_df['price_band'], failure_df['Result']))

plt.figure(figsize=(12, 6))
sns.countplot(data=failure_df, y='sector', hue='Result')
plt.title('Prediction Results by Sector - MLP (Primary)')
plt.tight_layout()
plt.savefig('../results/error_analysis_sector_dl_mlp_primary.png', bbox_inches='tight')
plt.show()

high_conf_fp = failure_df[
    (failure_df['Result'] == 'False Positive (Bad Bet)') & (failure_df['Probability'] >= 0.80)
]
high_conf_fn = failure_df[
    (failure_df['Result'] == 'False Negative (Missed Opp)') & (failure_df['Probability'] <= 0.20)
]

high_conf_fp[['sector', 'product', 'sales_price', 'Probability']].to_csv(
    '../results/high_confidence_false_positives_dl_mlp_primary.csv', index=False
)
high_conf_fn[['sector', 'product', 'sales_price', 'Probability']].to_csv(
    '../results/high_confidence_false_negatives_dl_mlp_primary.csv', index=False
)

print(f'High-confidence False Positives: {len(high_conf_fp)}')
if len(high_conf_fp) > 0:
    display(high_conf_fp[['sector', 'product', 'sales_price', 'Probability']].head(10))

print(f'High-confidence False Negatives: {len(high_conf_fn)}')
if len(high_conf_fn) > 0:
    display(high_conf_fn[['sector', 'product', 'sales_price', 'Probability']].head(10))"""

text_theory = """## 6. Theory Notes (AML + DL)
1. **SVM with RBF kernel**: maps inputs into a higher-dimensional space where nonlinear class boundaries can become linearly separable.
2. **Neural network forward pass**: each hidden layer computes weighted sums + bias and applies ReLU; output layer uses logistic behavior for binary probability.
3. **Activation functions**: ReLU improves gradient flow in hidden layers; sigmoid-like output maps scores to $[0, 1]$.
4. **Loss function**: binary cross-entropy (log-loss) penalizes wrong confident predictions more strongly.
5. **Optimizer**: Adam adaptively scales learning rates per parameter for stable and efficient training.
6. **Regularization**: L2 regularization (`alpha=0.001`) limits overfitting by penalizing large weights."""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(text_pipeline),
    nbf.v4.new_code_cell(code_pipeline),
    nbf.v4.new_markdown_cell(text_models),
    nbf.v4.new_code_cell(code_models),
    nbf.v4.new_markdown_cell(text_eval),
    nbf.v4.new_code_cell(code_eval),
    nbf.v4.new_markdown_cell(text_validation),
    nbf.v4.new_code_cell(code_validation),
    nbf.v4.new_markdown_cell(text_failure),
    nbf.v4.new_code_cell(code_failure),
    nbf.v4.new_markdown_cell(text_theory),
]

output_path = 'notebook/02_Modeling.ipynb'
with open(output_path, 'w') as f:
    nbf.write(nb, f)

print(f'Modeling Notebook generated at {output_path}')
