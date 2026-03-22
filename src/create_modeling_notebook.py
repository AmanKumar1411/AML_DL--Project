
import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# Predictive Modeling & Failure Analysis

## Goal
Build a model to predict `deal_stage` (Won/Lost) and perform a **rigorous failure analysis** to understand where the model (and potentially the sales process) falls short.

## Approach
1. **Feature Engineering**: Create features that capture the "quality" of the opportunity.
2. **Modeling**: Use Gradient Boosting for high performance.
3. **Failure Analysis**: Isolate incorrect predictions and find patterns."""

code_imports = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load Data
try:
    df = pd.read_csv("../data/processed/processed_sales_data.csv")
except FileNotFoundError:
    df = pd.read_csv("data/processed/processed_sales_data.csv")

print(f"Loaded data shape: {df.shape}")"""

text_feature_eng = """## 1. Feature Engineering & Selection
We must exclude features that are treated as "outcomes" (like `close_date`, `close_value`, `deal_duration`) to avoid **data leakage**."""

code_feature_eng = """# Define Features
# Exclude: 'deal_duration_days', 'close_value', 'log_close_value', 'close_year', 'close_month'
# Include: 'sector', 'year_established', 'revenue', 'employees', 'office_location', 
#          'series', 'sales_price', 'product', 'manager', 'regional_office', 
#          'engage_year', 'engage_month', 'log_revenue', 'log_employees'

features = [
    'sector', 'year_established', 'revenue', 'employees', 'office_location', 
    'series', 'sales_price', 'product', 'manager', 'regional_office', 
    'engage_year', 'engage_month', 'log_revenue', 'log_employees'
]

target = 'target'

# Check for missing values in features
print("Missing values in features:")
print(df[features].isnull().sum())

# Drop missing or fill (already handled in prep, but good to be safe)
df = df.dropna(subset=features)

X = df[features]
y = df[target]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training shapes: {X_train.shape}, {y_train.shape}")
print(f"Testing shapes: {X_test.shape}, {y_test.shape}")"""

text_model = """## 2. Model Training (Gradient Boosting)
We use a Pipeline to handle preprocessing (OneHotEncoding for categoricals) and Modeling."""

code_model = """# specific categorical columns
categorical_features = ['sector', 'office_location', 'series', 'product', 'manager', 'regional_office']
numeric_features = ['year_established', 'revenue', 'employees', 'sales_price', 'engage_year', 'engage_month', 'log_revenue', 'log_employees']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Gradient Boosting Classifier
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))])

clf.fit(X_train, y_train)
print("Model trained.")"""

text_eval = """## 3. Evaluation
Check accuracy, precision, recall, and ROC-AUC."""

code_eval = """y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()"""

text_failure = """## 4. Failure Analysis
This is the critical step. We analyze **where** the model failed.
We will isolate the False Positives (Predicted Won, Actual Lost) and False Negatives (Predicted Lost, Actual Won)."""

code_failure = """# Create a Failure DataFrame
failure_df = X_test.copy()
failure_df['Actual'] = y_test
failure_df['Predicted'] = y_pred
failure_df['Probability'] = y_prob

# Define Errors
failure_df['Result'] = 'Correct'
failure_df.loc[(failure_df['Actual']==1) & (failure_df['Predicted']==0), 'Result'] = 'False Negative (Missed Opp)'
failure_df.loc[(failure_df['Actual']==0) & (failure_df['Predicted']==1), 'Result'] = 'False Positive (Bad Bet)'

# Visualize Error Distribution by Sector
plt.figure(figsize=(12, 6))
sns.countplot(data=failure_df, y='sector', hue='Result')
plt.title("Prediction Results by Sector")
plt.show()

# High Confidence Failures
high_conf_fp = failure_df[(failure_df['Result'] == 'False Positive (Bad Bet)') & (failure_df['Probability'] > 0.8)]
print(f"High Confidence False Positives (Model was >80% sure but lost): {len(high_conf_fp)}")
if len(high_conf_fp) > 0:
    print(high_conf_fp[['sector', 'product', 'sales_price', 'Probability']].head())

high_conf_fn = failure_df[(failure_df['Result'] == 'False Negative (Missed Opp)') & (failure_df['Probability'] < 0.2)]
print(f"High Confidence False Negatives (Model was <20% sure but won): {len(high_conf_fn)}")
if len(high_conf_fn) > 0:
    print(high_conf_fn[['sector', 'product', 'sales_price', 'Probability']].head())
"""

text_feature_imp = """## 5. What Drives Predictions? (Feature Importance)"""

code_feature_imp = """# feature importance
# Accessing feature names is tricky with Pipeline + OneHotEncoder
# We make a best effort to map them

ohe = clf.named_steps['preprocessor'].named_transformers_['cat']
cat_feature_names = ohe.get_feature_names_out(categorical_features)
feature_names = numeric_features + list(cat_feature_names)

importances = clf.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[-20:] # Top 20

plt.figure(figsize=(10, 8))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(text_feature_eng),
    nbf.v4.new_code_cell(code_feature_eng),
    nbf.v4.new_markdown_cell(text_model),
    nbf.v4.new_code_cell(code_model),
    nbf.v4.new_markdown_cell(text_eval),
    nbf.v4.new_code_cell(code_eval),
    nbf.v4.new_markdown_cell(text_failure),
    nbf.v4.new_code_cell(code_failure),
    nbf.v4.new_markdown_cell(text_feature_imp),
    nbf.v4.new_code_cell(code_feature_imp)
]

output_path = "notebook/02_Modeling.ipynb"
with open(output_path, 'w') as f:
    nbf.write(nb, f)

print(f"Modeling Notebook generated at {output_path}")
