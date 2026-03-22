
import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# Exploratory Data Analysis: Predictive Sales

## Goal
Identify **non-obvious patterns** in the sales data that can inform our modeling strategy. We are looking for the "feel" of the data - what makes a deal close?

## Key Questions
1. **Price Sensitivity**: Do deep discounts lead to wins, or are they a sign of desperation?
2. **Seasonality**: Is there a "best time" to engage?
3. **Agent Performance**: What distinguishes top performers?
4. **Sector/Product Fit**: Where do we have the strongest competitive advantage?
"""

code_imports = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

os.makedirs("../results", exist_ok=True)

# Load Processed Data
try:
    df = pd.read_csv("../data/processed/processed_sales_data.csv")
    print("Data loaded successfully.")
except FileNotFoundError:
    df = pd.read_csv("data/processed/processed_sales_data.csv")
    print("Data loaded from alternate path.")

df.head()"""

text_univariate = """## 1. Target Variable: Deal Stage (Win/Loss)
Let's see the baseline win rate."""

code_target = """win_rate = df['target'].mean()
print(f"Overall Win Rate: {win_rate:.1%}")

plt.figure(figsize=(6, 6))
df['deal_stage'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
plt.title("Distribution of Deal Outcomes")
plt.savefig("../results/eda_win_rate_pie.png", bbox_inches='tight')
plt.show()"""

text_bivariate = """## 2. Sector and Product Analysis
Which sectors are we strongest in? Which products sell best?"""

code_sector_product = """fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Sector Win Rates
sector_win = df.groupby('sector')['target'].mean().sort_values(ascending=False)
sns.barplot(x=sector_win.index, y=sector_win.values, ax=axes[0], palette="Blues_d")
axes[0].set_title("Win Rate by Sector")
axes[0].set_ylabel("Win Rate")
axes[0].tick_params(axis='x', rotation=45)

# Product Win Rates
prod_win = df.groupby('product')['target'].mean().sort_values(ascending=False)
sns.barplot(x=prod_win.index, y=prod_win.values, ax=axes[1], palette="Greens_d")
axes[1].set_title("Win Rate by Product")
axes[1].set_ylabel("Win Rate")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("../results/eda_sector_product_win_rates.png", bbox_inches='tight')
plt.show()"""

text_discount = """## 3. The "Feel" of the Data: Discounting Strategy
**Hypothesis**: Lowering the price (deep discounts) increases the probability of winning.
**Analysis**: Compare `sales_price` (Suggested) vs `close_value` (Actual)."""

code_discount = """# Calculate Discount Percentage
# Note: close_value might be lower than sales_price.
# We treat 0 close_value as 'Lost' (which they are), but valid 'Won' deals should have non-zero.
# Let's filter for where we have valid comparisons (Won deals mainly, or non-zero close values)
# Actually, for lost deals in this dataset, close_value is 0?
# Let's check the data dictionary or just inspect.
# If close_value is 0 for Lost, we can't calculate discount for Lost. 
# So we can only analyze discount for WON deals to see if they were heavily discounted.
# OR we can assume 'sales_price' was the offer.

# Let's see close_value for Lost deals
print("Mean Close Value for Lost Deals:", df[df['target']==0]['close_value'].mean())

# Analysis:
# Since Lost deals have 0 close value, we can't see the "offered" price that failed.
# However, we CAN see if WON deals tend to be those with high discounts from the 'sales_price'.

won_deals = df[df['target'] == 1].copy()
won_deals['discount_pct'] = (won_deals['sales_price'] - won_deals['close_value']) / won_deals['sales_price']

plt.figure(figsize=(10, 6))
sns.histplot(won_deals['discount_pct'], bins=30, kde=True)
plt.title("Distribution of Discounts on WON Deals")
plt.xlabel("Discount Percentage (Positive = Price Cut)")
plt.axvline(0, color='red', linestyle='--')
plt.savefig("../results/eda_discount_dist.png", bbox_inches='tight')
plt.show()

print(f"Average Discount on Won Deals: {won_deals['discount_pct'].mean():.1%}")

# Does higher discount correlate with shorter deal duration?
sns.scatterplot(data=won_deals, x='discount_pct', y='deal_duration_days', alpha=0.3)
plt.title("Discount vs Deal Duration")
plt.savefig("../results/eda_discount_vs_duration.png", bbox_inches='tight')
plt.show()"""

text_seasonality = """## 4. Seasonality
Is there a 'golden month' to engage?"""

code_seasonality = """month_win = df.groupby('engage_month')['target'].mean()

plt.figure(figsize=(10, 5))
month_win.plot(marker='o')
plt.title("Win Rate by Engagement Month")
plt.ylabel("Win Rate")
plt.xticks(range(1, 13))
plt.grid(True)
plt.savefig("../results/eda_seasonality.png", bbox_inches='tight')
plt.show()"""

text_agents = """## 5. Agent Performance
Who are the "Closers"?"""

code_agents = """agent_perf = df.groupby('sales_agent').agg(
    deals=('target', 'count'),
    wins=('target', 'sum'),
    win_rate=('target', 'mean'),
    total_rev=('close_value', 'sum')
).sort_values('total_rev', ascending=False)

plt.figure(figsize=(12, 6))
sns.scatterplot(data=agent_perf, x='deals', y='win_rate', size='total_rev', hue='total_rev', sizes=(50, 500))
plt.title("Agent Performance: Volume vs Win Rate (Size = Revenue)")
# Label top 5 agents
for i in range(5):
    plt.text(agent_perf.iloc[i]['deals']+2, agent_perf.iloc[i]['win_rate'], agent_perf.index[i])
plt.savefig("../results/eda_agent_performance.png", bbox_inches='tight')
plt.show()

agent_perf.head(10)"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(text_univariate),
    nbf.v4.new_code_cell(code_target),
    nbf.v4.new_markdown_cell(text_bivariate),
    nbf.v4.new_code_cell(code_sector_product),
    nbf.v4.new_markdown_cell(text_discount),
    nbf.v4.new_code_cell(code_discount),
    nbf.v4.new_markdown_cell(text_seasonality),
    nbf.v4.new_code_cell(code_seasonality),
    nbf.v4.new_markdown_cell(text_agents),
    nbf.v4.new_code_cell(code_agents)
]

output_path = "notebook/01_EDA.ipynb"
with open(output_path, 'w') as f:
    nbf.write(nb, f)

print(f"EDA Notebook generated at {output_path}")
