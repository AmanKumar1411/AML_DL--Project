# Phase 1 Plan — Honest Benchmark (measure only)

Status: **approved and executed** on branch `phase1-benchmark`. The results are in the README's "Benchmark (Phase 1)" section. Deviations from this plan:
- **Environment:** Option B was chosen (tabpfn 9.2.0), and v1 reproduced exactly on it.
- **Ceiling:** TabPFN-3.5 was added once access was set up (gated Hugging Face weights, a Prior Labs licence token, and root certificates for python.org Python). Until then, CatBoost stood in as a temporary ceiling; it is now kept as a classical baseline.
- **Requirements:** `Requirements.txt` was renamed to `requirements.txt` with exact pins, instead of adding a separate `requirements-benchmark.txt`.

The rest of this document is the original plan, unchanged.

## 0. Summary

- **Your suspicion is confirmed, to 10 decimals.** The README's "optimal threshold 0.3 → F1 0.774, recall 1.0" is exactly the score of a classifier that predicts "won" for every row. That threshold was also picked on the test set.
- At the default 0.5 threshold, v1 is barely different. The DualPathMLP predicts "won" for 97% of test rows and the Hybrid for 92%. Both have **lower accuracy than always guessing "won"** (0.625 and 0.604 vs 0.631). All three v1 ROC-AUCs (0.514–0.521) fall inside the 95% interval for chance on this test set, [0.468, 0.532].
- **There is one blocker before Stage 3: the data is not on this machine.** `data/` is gitignored and absent. No copy of `sales_pipeline.csv` exists anywhere under your home folder. Every fact below comes from the code plus the outputs saved inside the notebooks.
- **Five decisions are needed from you** (Section 7): approve the plan, pick an environment, supply the data, accept the TabPFN licence, and choose a README correction style.

---

## 1. Findings (Stage 1)

### 1.1 Repo map

| Path | What it is |
|---|---|
| `main.py` | 5 lines. Calls `run_phase3_pipeline()`. |
| `src/process_data.py` | `load_and_process_data()` turns the raw CSVs into `data/processed/processed_sales_data.csv`. It filters to Won/Lost, builds the target, left-joins accounts/products/teams, adds date parts and `log1p` features, and fills NaNs in `series` and `sales_price`. |
| `src/phase3_pipeline.py` (729 lines) | **The v1 system.** Feature lists (`FEATURES`, `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`). `prepare_data()` does the 80/20 split. `build_ml_model()` builds the LogReg pipeline. `DualPathMLP` is a NumPy MLP. `split_numeric_categorical()` holds the MLP's scaler and encoder. `fit_models()`, `evaluate_predictions()`, `run_hybrid_model()`, `run_ablation_study()`, `run_threshold_analysis()`, `run_shap_analysis()`, `summarize_misclassifications()`, `save_architecture_diagram()`. |
| `src/create_eda_notebook.py`, `src/create_modeling_notebook.py` | One-off generators that write `01_EDA.ipynb` and `02_Modeling.ipynb`. They need `nbformat`, which is not in requirements. |
| `src/__init__.py` | Empty. Makes `python -m src.…` work. |
| `notebook/data_loading_and_cleaning.ipynb` | Exploratory notebook (71 cells). Duplicates `process_data.py`, then runs LR, RF, XGBoost and synthetic-"conversation text" experiments. |
| `notebook/01_EDA.ipynb` | Charts of win rate, sector/product, seasonality and agents → `results/eda_*.png`. |
| `notebook/02_Modeling.ipynb` | LR, SVM-RBF and DualPathMLP on the same 80/20 split, plus confusion matrices and SHAP. `RUN_CV = False`. |
| `notebook/03_Results_Analysis.ipynb` | XGBoost comparison. **No saved outputs** (never run as committed). Needs `xgboost`. |
| `reports/*.pdf` | Course reports. |
| `results/` | 31 committed outputs (charts, CSVs, one TXT). Producers are listed in 1.7. |
| `Requirements.txt` | Pins numpy 1.26.4, pandas 2.2.2, sklearn 1.4.2, scipy 1.11.4, matplotlib 3.8.4, seaborn 0.13.2, shap 0.44.1, torch 2.3.1, jupyter. |
| `Readme.md` | Project README. Section 1.5 lists the claims the benchmark contradicts. |

### 1.2 Data flow

```
data/raw/{accounts,products,sales_pipeline,sales_teams}.csv        ← NOT PRESENT (gitignored)
   │  src/process_data.py: load_and_process_data()   (CWD-relative paths)
   ▼
data/processed/processed_sales_data.csv   6,711 rows × 27 cols
   │  phase3_pipeline.load_dataset()  (calls process_data if the file is missing)
   │  prepare_data(): dropna(FEATURES+target) → drops 0 rows
   │                  train_test_split(0.2, stratify=y, random_state=42) → 5,368 / 1,343
   ▼
fit_models():  LogReg Pipeline.fit(train)
               StandardScaler + OneHotEncoder fit(train) → inner split 80/20 of train
               DualPathMLP.fit(inner-train, early-stop on inner-val)
   ▼
probabilities on the 1,343 test rows → hybrid = 0.4·LR + 0.6·MLP
   ▼
results/hybrid_model_comparison.csv, ablation_study_*.{csv,png}, threshold_analysis*.{csv,png},
        shap_*.{png,csv,txt}, error_segmentation_*.{png,csv}, architecture_diagram.png
```

### 1.3 Facts

**Target and rows.**
- The raw `sales_pipeline.csv` has 8,800 opportunities: Won 4,238, Lost 2,473, Engaging 1,589, Prospecting 500.
- The 2,089 open deals (Engaging + Prospecting) are dropped because their outcome is unknown. That leaves 6,711 rows, with target = 1 for Won and 0 for Lost.
- The merges are left joins and keep all 6,711 rows. `dropna` removes nothing after the fills.
- **1,343 is the old test split, not the dataset size.**

**Class balance.**
- p = 4,238 / 6,711 = **0.6315** (63.2% won).
- The old test split has 848 won and 495 lost (p = 0.6314).

**Features (14).**
- 8 numeric: `year_established, revenue, employees, sales_price, engage_year, engage_month, log_revenue, log_employees`.
- 6 categorical: `sector` (10), `office_location` (15), `series` (4, including "Unknown"), `product` (7), `manager` (6), `regional_office` (3). These become 45 one-hot columns.
- Correctly excluded: `close_value` (it is 0 for every lost deal, a direct target leak) and `deal_duration_days` / `close_*` (only known after the outcome).
- Also excluded: `sales_agent`, `account`, `subsidiary_of`.
- Every numeric feature has |r| ≤ 0.013 with the target, except `engage_year` (r = −0.085).

**Preprocessing and where it is fit.**
- Before the split, on all 6,711 rows: `log1p` transforms, `series` NaN → "Unknown", and `sales_price` NaN → **median over all rows** (`process_data.py:63`).
- There are 1,147 NaNs (17%). They very likely come from a product-name mismatch: `GTXPro` in the pipeline table vs `GTX Pro` in the products table. This is inferred and needs the CSVs to confirm.
- After the split, on train only: LogReg uses `ColumnTransformer(StandardScaler, OneHotEncoder(handle_unknown='ignore'))` inside its Pipeline. The MLP has its own `StandardScaler` and dense `OneHotEncoder`, both fit on the full 5,368-row train set.

**DualPathMLP** (`phase3_pipeline.py:122-347`). It is **pure NumPy** with hand-written forward, backprop and Adam. **torch is never imported anywhere in the repo.**
- Numeric path: Dense(8→16) + ReLU. Categorical path: Dense(45→32) + ReLU.
- Concat(48) → Dense(64) + ReLU → Dense(32) + ReLU → Dense(1) + sigmoid.
- He-normal initialisation and zero biases, from `np.random.default_rng(42)`. The same generator shuffles the minibatches.
- Loss: mean BCE + 0.5·α·ΣW², with α = 0.001. No class weighting.
- Optimiser: Adam, learning rate 1e-3, β = (0.9, 0.999), batch size 128.
- Up to 220 epochs (350 in the notebook). Early stopping: patience 20 and min-delta 1e-6 on the full-batch validation loss. The best weights are restored.
- **Early stopping watches an inner 20% stratified split of the training set** (4,294 / 1,074 rows, `random_state=42`), **not the test set.** ✓
- The notebook log shows it stopped at epoch 46, so the best epoch was 26. The last logged validation losses are 0.742–0.749. A constant-p predictor scores 0.658.

**Blend and threshold.**
- `hybrid = 0.4·LR_prob + 0.6·MLP_prob` (`:452`). The weights are fixed constants, not tuned.
- The LR uses `class_weight='balanced'`, so its probabilities centre near 0.5. The MLP is unweighted, so its probabilities centre near p. The blend averages two differently calibrated scores.
- The headline table uses a 0.5 threshold.
- `run_threshold_analysis` (`:486-526`) scores thresholds {0.3, 0.4, 0.5, 0.6, 0.7} **on `y_test`** and takes the max F1. `idxmax` breaks ties toward the first value, so it picks 0.3.

### 1.4 Test of your suspicion

F1 of an always-positive classifier = 2p / (1 + p).

| | p | always-positive F1 | Reported |
|---|---|---|---|
| Full data | 0.631501 | 0.774135 | — |
| Old test split | 848/1343 = 0.6314221891 | **0.7740757645** | Hybrid @ 0.3: F1 **0.7740757645**, precision **0.6314221891**, recall **1.0** |

The reported numbers match to every printed digit. Precision equals the base rate and recall is 1.0. **At 0.3 (and at 0.4) the Hybrid predicts "won" for all 1,343 rows.** The same F1 value also shows up in the cleaning notebook's text model and its "tuned XGBoost" (0.7738), which means it is the number any model hits by predicting "won" for everything.

At 0.5 (confusion matrices rebuilt from the saved FP/FN counts):

| Model | TP | FP | TN | FN | Share predicted "won" | Accuracy | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Always "won" | 848 | 495 | 0 | 0 | 100% | **0.631** | 0.500 |
| DualPathMLP | 825 | 481 | 14 | 23 | 97.2% | 0.625 | 0.521 |
| Hybrid | 777 | 461 | 34 | 71 | 92.2% | 0.604 | 0.520 |
| LogReg (balanced) | 317 | 176 | 319 | 531 | 36.7% | 0.474 | 0.514 |

With 848 positives and 495 negatives, the standard error of an AUC near 0.5 is about 0.016 (Hanley–McNeil). The 95% interval under chance is [0.468, 0.532]. **None of the v1 models can be told apart from a coin flip on this split.**

### 1.5 Problems found

| # | Severity | Where | Problem |
|---|---|---|---|
| P1 | High | `phase3_pipeline.py:486-526` | **Threshold tuned on the test set.** The "optimal 0.3" was picked on `y_test`, and it is the always-positive classifier. |
| P2 | High | README, `hybrid_model_comparison.csv` | **No floor reported.** F1 and recall look good only because the models predict "won" for almost every row. Accuracy is below the majority-class rate, and AUC is within noise of 0.5. |
| P3 | Medium | `:476-481` | The **ablation numbers are artifacts of a fixed threshold.** "Removing DL reduces recall by 59.2%" compares recall at 0.5 between models with different calibration. "Hybrid achieves best balance — proves necessity of both" is printed **unconditionally**, whatever the numbers say. |
| P4 | Medium | `:424` | Comparison tables are **sorted by F1**, which rewards predicting the majority class. |
| P5 | Medium | `prepare_data` | **One 80/20 split, no CV, no variance estimate** (`RUN_CV = False` in the notebook). |
| P6 | Low | `process_data.py:63` | **Imputation fit before the split.** The `sales_price` median is computed over all rows. The leak is tiny (it is a list price), but it is pre-split. The likely root cause is the `GTXPro`/`GTX Pro` name mismatch, which wipes the real price and series for 17% of rows. Fix in Phase 2. |
| P7 | Low | `:452` | The blend averages a balanced-weight LR with an unweighted MLP, so the two probability scales differ. |
| P8 | Info | splits | **Account overlap.** 6,711 deals come from 85 accounts. A random split puts the same account in train and test, and the account-level features (sector, revenue, employees, year, office) nearly identify the account. Scores measure performance on accounts the model has already seen. Phase 2 could add a GroupKFold-by-account check. |
| P9 | Info | features | `engage_year` (the only numeric feature with any correlation) is **probably a right-censoring artifact.** Only closed deals are kept and won deals take longer, so late-engaged deals that had already closed lean "lost". Phase 2. |
| P10 | Low | `process_data.py:14-23,66`, notebooks, README | **Hardcoded and CWD-relative paths.** `data/raw/…` with a `../` fallback only works from the repo root or `notebook/`. Notebooks write to `../results`. The README's setup uses `/Users/amankumar/Desktop/predictive-sales project`, which does not exist. `phase3_pipeline.py` correctly uses `PROJECT_ROOT`. |
| P11 | Low | everywhere | **Seeds:** deterministic in practice (`rng(42)` and `random_state=42` everywhere). But there is no global seeding, no saved split indices and no version log. |
| P12 | Medium | env | **Environment drift.** No interpreter here has the pinned versions, and there is no venv. `shap` is missing, so `main.py` currently fails at import. `torch` is pinned but unused. `xgboost` and `nbformat` are used but not listed. |
| P13 | Medium | README | **Wrong or misleading claims:** "Training dataset (1,343 rows)" (that's the test split); "Optimal: 0.3 (F1 0.774, Recall 1.0)" (P1); the ablation finding (P3); "Preprocessing fitted only on training data" (not true for P6 or the threshold); and the SHAP and error-segment "insights", which come from a model at chance level. |
| — | OK | `evaluate_predictions` | The metric arithmetic itself is correct (standard sklearn calls, AUC on probabilities). The problems are in what was chosen and how it was reported, not the formulas. |

Two other notes:
- `data_loading_and_cleaning.ipynb` cell 31 builds a synthetic `conversation_text` from `row['target']`, which is direct target leakage. It is not in the v1 path, and later cells redefine it.
- The notebook XGBoost "AUC 0.63–0.66" used `deal_duration_days`, `close_month` and `sales_agent`. The first two are only known after the outcome.

### 1.6 Environment

| | |
|---|---|
| Hardware | Apple M2, 8 cores, **8 GB RAM**, arm64. No CUDA. The Apple GPU (MPS) is the only accelerator. |
| `python3` on PATH | 3.14.7 (Homebrew), **no packages installed**. `python` is not on PATH. |
| Python 3.12.0 (python.org, `/usr/local/bin/python3`) | numpy 2.3.4, pandas 2.3.3, sklearn 1.7.2, scipy 1.16.3, matplotlib 3.10.7, seaborn 0.13.2, xgboost 3.2.0 |
| Not installed anywhere | **torch, shap, catboost, tabpfn** |
| On PyPI today | tabpfn 9.2.0 (default model TabPFN-3.5), catboost 1.2.10, torch 2.15.0, shap 0.54.1 |
| Pip dry-run (nothing installed) | `Requirements.txt` + tabpfn + catboost **resolves cleanly**, but to **tabpfn 6.4.1**: 9.x needs `torch>=2.5` and the file pins 2.3.1. The unpinned stack resolves to tabpfn 9.2.0, torch 2.15, numpy 2.5.4, pandas 3.1.1 and sklearn 1.9.1. |

### 1.7 Files that look obsolete (not deleting; your call)

- `results/confusion_matrix_dl_mlp_primary.png`, `error_analysis_sector_dl_mlp_primary.png`, `confusion_matrix_ml_legacy_gradient_boosting.png`, `feature_importance_ml_legacy_gradient_boosting.png` and `error_analysis_sector_ml_legacy_gradient_boosting.png`: **no code produces these anymore.**
- `src/create_eda_notebook.py` and `src/create_modeling_notebook.py`: one-off notebook generators. The notebooks are now the working copies.
- `notebook/03_Results_Analysis.ipynb`: never executed as committed.
- `notebook/data_loading_and_cleaning.ipynb`: duplicates `process_data.py` and contains the leaky experiments above.

---

## 2. Design

### 2.1 Folds

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the same 6,711 rows v1 uses (after v1's `dropna`).
- The folds are saved to `results/benchmark_v1_folds.csv` with columns `row_index, opportunity_id, fold`.
- Each run regenerates them and **stops with an error if they differ from the saved file**, so Phases 2–3 are scored on identical folds.
- Every model sees exactly these folds.

### 2.2 No-leak rules

For each fold:
- Everything is fit on that fold's training part only. This includes v1's own scaler and encoders, which are reused unchanged.
- Anything tuned uses an **inner split of the training part**. That split is v1's own `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
- The test fold is touched exactly once, to score.

### 2.3 Models (all as-is)

| Label | Build |
|---|---|
| **Dummy** | `DummyClassifier(strategy="most_frequent")`. It outputs P(won) = 1.0 for every row. |
| **LogReg v1** | `phase3_pipeline.build_ml_model()`, unchanged. |
| **DualPathMLP v1** | v1's `split_numeric_categorical()` fit on the training fold, then the inner split, then `DualPathMLP(..., random_state=42)` with default hyperparameters (220 epochs, patience 20), early-stopped on the inner validation set. This is the same code path as `fit_models()`. |
| **Hybrid v1** | `0.4·LogReg + 0.6·MLP`, both trained inside the fold. Headline metrics use threshold 0.5. **Also reported:** "F1 @ tuned threshold (inner-val)", using v1's grid {0.3…0.7}, picking max F1 with ties to the lowest, as v1 does. On the inner validation set, LR is refit on the inner-train rows only, so its validation probabilities are out-of-sample. The MLP is used as trained. It early-stopped on that same inner validation set, which makes the tuned threshold slightly optimistic, but it never touches the test fold. |
| **TabPFN (ceiling)** | `TabPFNClassifier()` with default settings. The only non-defaults are `random_state=42` (if the constructor has it) and `device="cpu"` (for determinism). Input is the same 14 raw features. If TabPFN rejects strings, the categoricals are ordinal-encoded inside the fold. |
| *CatBoost (temporary ceiling)* | **Only if TabPFN is unavailable.** `CatBoostClassifier(random_seed=42, verbose=0, cat_features=<6 cols>, allow_writing_files=False)`, otherwise default. Clearly labelled. |

### 2.4 Metrics

Per fold:
- ROC-AUC
- PR-AUC (`average_precision_score`)
- F1, precision, recall and accuracy at 0.5
- Brier score

Plus two diagnostics:
- the share predicted "won" at 0.5
- the test-fold positive rate

For the Hybrid only, also: the tuned threshold and the F1 at that threshold, labelled as such.

The summary reports mean ± std over folds (sample std, ddof = 1).

**Built-in sanity checks.** The run stops if any of these fail:
- The Dummy's ROC-AUC is exactly 0.5 in every fold.
- The Dummy's PR-AUC equals that fold's positive rate (to within 1e-9).
- Every fold's positive rate is within ±0.002 of p.

Note on the Brier score: because `most_frequent` outputs 1.0, the Dummy's Brier is 1 − p ≈ 0.368. The fairer no-skill reference is a forecast of the constant p, which scores p(1 − p) ≈ 0.233. I'll print that as a reference line rather than add a sixth model.

### 2.5 Reproducibility

- `set_seeds(42)` seeds `random`, NumPy's global generator and `torch.manual_seed`, and sets `torch.use_deterministic_algorithms(True)`. Every model also gets its own explicit seed.
- Fold indices are saved.
- `results/benchmark_v1_env.json` records Python, OS, CPU, all package versions, and the TabPFN model version and device.
- Timings are printed to the console only, so the two runs can be compared byte for byte.
- The full benchmark runs twice and I diff the CSVs.

**Check before the CV run.** I'll call the v1 functions in memory on v1's original 80/20 split and confirm I get back `hybrid_model_comparison.csv` (MLP AUC 0.5206 / F1 0.7660, LR 0.5140 / 0.4728, Hybrid 0.5196 / 0.7450). This proves the benchmark wraps v1 faithfully. **Nothing in `results/` from v1 is overwritten.**

### 2.6 Interface

```python
# src/benchmark.py
def evaluate_model(model_factory: Callable[[], ProbaClassifier],
                   X: pd.DataFrame, y: np.ndarray,
                   folds: list[tuple[np.ndarray, np.ndarray]],
                   name: str) -> pd.DataFrame:      # one row per fold
```

- `ProbaClassifier` means sklearn-style `fit(X, y)` / `predict_proba(X)`. Small adapter classes wrap DualPathMLP, the Hybrid and TabPFN to match it.
- A model that exposes `tuned_threshold_` also gets the tuned-F1 columns.
- Phases 2–3 can pass any new factory with the same folds.
- CLI: `python -m src.benchmark --out results/benchmark_v1.csv [--models dummy,logreg,mlp,hybrid,tabpfn] [--ceiling tabpfn|catboost]`.

### 2.7 Outputs

- `results/benchmark_v1.csv`: one row per fold × model.
- `results/benchmark_v1_summary.csv`: mean ± std per model.
- `results/figures/benchmark_v1_ladder.png`: two panels (ROC-AUC, PR-AUC) of horizontal bars, ordered Dummy → LogReg → DualPathMLP → Hybrid → TabPFN, with std error bars and a dashed line at the Dummy level (0.5 and ≈0.632).

### 2.8 README

Add a **"Benchmark (Phase 1)"** section with the summary table, the chart, the class balance (63.2% won, 6,711 deals) and 3 honest bullets.

Then fix every claim listed under P13 **in place**, each with a short "corrected in Phase 1" note:
- the 1,343-row claim
- the threshold-0.3 claim
- the ablation finding
- the "fit only on training data" claim
- the old metrics table (labelled "single 80/20 split, superseded")
- the SHAP and segment caveat
- the hardcoded install path
- where to put the data

---

## 3. Files

| Action | File |
|---|---|
| **Create** | `src/benchmark.py` |
| **Create** | `requirements-benchmark.txt` (exact versions used; option A: `-r Requirements.txt` + tabpfn + catboost) |
| **Create** (generated) | `results/benchmark_v1.csv`, `results/benchmark_v1_summary.csv`, `results/benchmark_v1_folds.csv`, `results/benchmark_v1_env.json`, `results/figures/benchmark_v1_ladder.png` |
| **Edit** | `Readme.md` only |
| **Not touched** | `main.py`, `src/phase3_pipeline.py`, `src/process_data.py`, notebooks, `Requirements.txt`, all existing `results/*` |
| Local only (gitignored) | `.venv/`, `data/raw/*.csv` (you supply these), `data/processed/processed_sales_data.csv` (generated by the existing `process_data.py`) |

---

## 4. Risks and fallbacks

**1. Data missing (blocker).**
- The repo expects `data/raw/{accounts,products,sales_pipeline,sales_teams}.csv`.
- The shapes in the notebook (8,800 opportunities, 85 accounts, 7 products, 35 agents) look like Maven Analytics' public **"CRM Sales Opportunities"** dataset. Use your original copy if you have one.
- Check: `sales_pipeline.csv` should load as (8800, 8). A different copy will give different numbers.

**2. TabPFN weights need your login and licence acceptance.** These steps are from the tabpfn 9.2.0 docs.
- The first `fit` opens a browser to log in at Prior Labs (https://ux.priorlabs.ai) and accept the licence. The token is cached after that.
- Headless alternative: accept under the **License** tab at ux.priorlabs.ai, then `export TABPFN_TOKEN=<token>`.
- **Licence:** the code is Apache-2.0. The TabPFN-2.5, 2.6, 3 and 3.5 weights are **non-commercial**. The TabPFN-2 weights are Apache-2.0 with an attribution requirement. A course project should qualify, but you have to accept the terms yourself; I can't.
- If we pin tabpfn 6.4.1 (option A), its login flow may be older (possibly Hugging Face based). I'll read the message the installed version prints and give you the exact steps.

**3. CPU limits and speed.**
- Training folds are about 5,369 rows. That is above TabPFN's CPU guard: 5,000 rows for v3.x models, 1,000 for older ones.
- So I'll set `TABPFN_ALLOW_CPU_LARGE_DATASET=true`. This is an execution guard, not a model setting.
- If CPU takes more than about 15 minutes per fold, I'll switch to MPS (`device="auto"`; 8 GB RAM; `TABPFN_MPS_MEMORY_FRACTION` defaults to 0.7) and re-check that runs are identical, since GPU kernels may not be deterministic.

**4. Hosted `tabpfn-client`.**
- It needs a Prior Labs account and token, and it **uploads the dataset to Prior Labs' servers**. I'd use it only with your explicit OK.
- I couldn't confirm its usage limits from the docs.

**5. TabPFN unavailable.**
- Fall back to CatBoost with default settings, labelled **"CatBoost (temporary ceiling)"**.
- TabPFN can be dropped in later on the same saved folds without rerunning the rest.

**6. Install problems.**
- If option A fails to install on Python 3.12.0 (the dry run says it resolves), I'll stop and come back to you.
- Option B is the alternative, with the v1 reproduction check as the safety net.

**7. Non-determinism.**
- Expected only from TabPFN on MPS.
- If the run-twice check fails for TabPFN alone, I'll report the size of the difference instead of hiding it.

---

## 5. Estimated runtime (M2, CPU)

| Step | Estimate |
|---|---|
| venv and installs (torch download, TabPFN weights on first use) | 3–8 min, once |
| `process_data` | < 10 s |
| v1 reproduction check | < 1 min |
| Dummy + LogReg, 5 folds | < 10 s |
| DualPathMLP + Hybrid, 5 folds (NumPy, ≤220 epochs × ~34 batches; the Hybrid trains its own components) | ~1–3 min |
| TabPFN, 5 folds, CPU | **the main unknown: roughly 2–10 min per fold → 10–50 min** |
| CatBoost (only if needed) | ~1–3 min |
| **One full run** | **~15–55 min** · **two runs ≈ 0.5–2 h** |

I'll time TabPFN on fold 1 first and tell you before starting the double run if it's at the slow end.

**Expectation (hypothesis, not a result).** LogReg, MLP and Hybrid will probably land at ROC-AUC ≈ 0.50–0.53 and PR-AUC ≈ 0.63–0.66, inside the Dummy's noise band. TabPFN may not clear the floor either. If it doesn't, the 14 v1 features carry almost no signal, and Phase 2 is a feature problem, not a model problem.

---

## 6. Planned Stage 3 order

1. Create `.venv` (Python 3.12.0) and install. Log versions.
2. You place the data → run `process_data` → verify 6,711 rows and p = 0.6315.
3. Run the v1 reproduction check on the old 80/20 split.
4. Write `src/benchmark.py` → run → check the sanity asserts.
5. Run a second time → diff the CSVs.
6. Draw the chart and edit the README.
7. Report: files changed, results table, 2–3 Phase 2 suggestions.

---

## 7. Decisions needed before I write code

1. **Approve or change** this plan.
2. **Environment:**
   - **A (recommended):** fresh `.venv` with your exact `Requirements.txt` pins, plus **tabpfn 6.4.1** and catboost 1.2.10. v1 runs on the versions it was built with. The pip dry run resolves cleanly. TabPFN's default model in 6.4.1 isn't verified yet; I'll log it.
   - **B:** fresh `.venv` with current versions: **tabpfn 9.2.0 / TabPFN-3.5** (the newest ceiling), torch 2.15, numpy 2.5, **pandas 3.1**, sklearn 1.9. Stronger ceiling, but v1 runs on newer libraries, and the reproduction check has to pass.
3. **Data:** put the 4 raw CSVs in `data/raw/`.
4. **TabPFN licence:** accept it at ux.priorlabs.ai (or let the first run open the browser). Or approve CatBoost as the temporary ceiling for now.
5. **README:** fix wrong claims in place with "corrected" notes (default), or move the old v1 numbers into a "Superseded (single split)" subsection.
