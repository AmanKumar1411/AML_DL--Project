
import pandas as pd
import numpy as np
import os

# Set working directory to project root if running from script
# Assuming script is in src/ or we run from root
# Let's assume we run from project root

def load_and_process_data():
    print("Loading data...")
    # Load datasets
    try:
        accounts = pd.read_csv("data/raw/accounts.csv")
        products = pd.read_csv("data/raw/products.csv")
        pipeline = pd.read_csv("data/raw/sales_pipeline.csv")
        sales_teams = pd.read_csv("data/raw/sales_teams.csv")
    except FileNotFoundError:
        # Try adjusting path if running from notebook dir
        accounts = pd.read_csv("../data/raw/accounts.csv")
        products = pd.read_csv("../data/raw/products.csv")
        pipeline = pd.read_csv("../data/raw/sales_pipeline.csv")
        sales_teams = pd.read_csv("../data/raw/sales_teams.csv")

    print("Data loaded. Processing...")
    
    # 1. Filter for Won/Lost
    pipeline = pipeline[pipeline['deal_stage'].isin(['Won', 'Lost'])].copy()
    
    # 2. Create Target
    pipeline['target'] = pipeline['deal_stage'].map({'Won': 1, 'Lost': 0})
    
    # 3. Merges
    master_df = pipeline.merge(accounts, how='left', on='account')
    master_df = master_df.merge(products, how='left', on='product')
    master_df = master_df.merge(sales_teams, how='left', on='sales_agent')
    
    # 4. Dates
    master_df['engage_date'] = pd.to_datetime(master_df['engage_date'])
    master_df['close_date'] = pd.to_datetime(master_df['close_date'])
    
    # 5. Features
    master_df['deal_duration_days'] = (master_df['close_date'] - master_df['engage_date']).dt.days
    
    # Log transforms (handling zeros/negatives with log1p)
    # Using np.log1p for numerical stability
    master_df['log_close_value'] = np.log1p(master_df['close_value'])
    master_df['log_revenue'] = np.log1p(master_df['revenue'])
    master_df['log_employees'] = np.log1p(master_df['employees'])
    
    # Date parts
    master_df['engage_year'] = master_df['engage_date'].dt.year
    master_df['engage_month'] = master_df['engage_date'].dt.month
    master_df['close_year'] = master_df['close_date'].dt.year
    master_df['close_month'] = master_df['close_date'].dt.month

    # Missing values handling (from notebook)
    # Convert 'subsidiary_of' is dropped in notebook for modeling, but we might want it for EDA? 
    # The notebook drops it for model_df. I'll keep it in master_df but fill na for others.
    # Actually, for consistency with the user's notebook logic for modeling, I'll apply the fills.
    
    master_df['series'] = master_df['series'].fillna('Unknown')
    master_df['sales_price'] = master_df['sales_price'].fillna(master_df['sales_price'].median())
    
    # Construct output path
    output_dir = "data/processed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, "processed_sales_data.csv")
    master_df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")
    print(f"Shape: {master_df.shape}")
    
    # Also save the modeling version (dropping columns) if needed?
    # The user wants "EDA in 01_EDA" and "Modeling in 02_Modeling".
    # EDA needs the rich data. Modeling needs the clean numeric/encoded data.
    # I'll let 02_Modeling do the final dropping/encoding to keep it self-contained.
    
    return master_df

if __name__ == "__main__":
    load_and_process_data()
