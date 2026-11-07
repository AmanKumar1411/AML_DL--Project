# Archived notebooks (history only)

These notebooks are kept for the project's history. **Do not use their numbers.** They contain leaky experiments:

- **`data_loading_and_cleaning.ipynb`**
  - Trains on `deal_duration_days`, `close_year` and `close_month`. These are only known after a deal closes.
  - One version builds a synthetic `conversation_text` from the label (`row['target']`).
  - Picks decision thresholds on the test set.
- **`03_Results_Analysis.ipynb`**
  - The "feature engineering" model drops only `target` and `deal_stage`. It therefore trains on `close_value`, which is 0 for every lost deal (a direct label leak), along with the other close-date fields.
  - Picks the F1 threshold on the test set.

The current, leak-free evaluation is `python -m src.benchmark` (see the Benchmark section of the main README).
