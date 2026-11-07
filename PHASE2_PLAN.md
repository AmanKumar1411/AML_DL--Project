# Phase 2 Plan: Feature Engineering

Status: **Stage 1 (analysis) and Stage 2 (plan) done. Stage 3 (implementation) not started.**
Branch: `overnight-upgrade`. Model architectures and hyperparameters stay unchanged (that is Phase 3).

Every number below can be reproduced with a read-only script (it writes nothing):

```bash
python scripts/phase2_stage1_analysis.py                     # sections 1-4, ~1 min
python scripts/phase2_stage1_analysis.py --catboost-preview  # + section 5, ~3 min
```

## 0. Summary

- **Step 0 was already done** (commits `1f2b172`, `d3bd0d1`). TabPFN-3.5 is the Phase 1 ceiling: ROC-AUC 0.567 ± 0.010, Brier 0.229. CatBoost is kept as a classical baseline. Tonight's 50-row smoke test still loads and fits on CPU (4.9 s).
- **The static attributes carry no detectable signal.** Agent, manager, office, product, account, sector, price and every interaction tested match chance. χ² p is between 0.22 and 0.97 across 15 tables. Out-of-fold AUC is 0.49–0.51.
- **The only signal is when the deal was engaged, specifically its position within the quarter.**
  - Deals engaged 14–41 days before quarter end win 69–74%. Deals engaged in the quarter's first month win 55–57% (2017 Q2–Q4).
  - A target encoding of weeks-to-quarter-end alone scores OOF AUC 0.565. That roughly equals TabPFN's Phase 1 ceiling with all 14 features (0.567).
  - With every date feature removed, CatBoost drops to 0.488.
- **`engage_year` is a data-window artifact. Recommendation: drop it.**
  - The data only contains deals closed on or after 2017-03-01. So a deal engaged in late 2016 appears only if it was still open in March, which favours won deals.
  - The 2016 cohort wins 82%, against 62% for 2017.
  - On deals engaged after 2017-03-01, `engage_year` scores exactly AUC 0.500.
  - `engage_month` carries the same artifact on top of the real within-quarter signal. The plan replaces it with a within-quarter encoding.
- **History features will most likely not help.**
  - Time-aware agent, account and product win rates score AUC 0.47–0.49.
  - Their deal counts score 0.46–0.47, only because a count is a proxy for time.
  - They stay in the ladder as clearly labelled negative controls, so the README can say with evidence that they did not help.
- **Account overlap is total:** all 85 accounts appear in every train and test fold. GroupKFold by account is therefore the important secondary check.
- **Expected outcome: the best AUC will probably stay below 0.58.** A CatBoost preview on the planned F2 rung scores 0.552 on all rows, 0.554 on deals engaged after 2017-03-01 (F0: 0.542 / 0.519). If that holds, the README will say so plainly.

---

## 1. Stage 1 findings

### 1.1 Step 0: TabPFN in the Phase 1 benchmark

| Sub-step | Status |
|---|---|
| a. TabPFN-3.5 loads, 50-row fit | Done before (Phase 1), re-checked tonight: fit + predict in 4.9 s on CPU |
| b. Fold timing | Done: about 2 min per fold on M2 CPU, so MPS is not needed |
| c. Re-run with the saved folds | Done: the other five models' per-fold numbers are bit-identical to before, and two full runs gave byte-identical files |
| d. README table and ladder chart | Done: TabPFN shown as the ceiling, CatBoost as the classical baseline, plus a constant-forecast Brier row |
| e. Fallback | Not needed |

### 1.2 Join audit: unmatched names

| Join | Unmatched name | Rows affected (all / closed) | Effect in v1 |
|---|---|---|---|
| `product` → products.csv | **`GTXPro`** (table says `GTX Pro`) | **1,480 / 1,147** (17.1% of closed deals) | `series = "Unknown"` and `sales_price = 1,096` (the all-row median, computed before any split; it happens to equal GTX Plus Basic's price). The true values are GTX and 4,821. The `product` category itself is intact, so the deal is still identifiable. |
| `account` → accounts.csv | none | 1,425 rows have no account, **all of them open deals** (Engaging 1,088, Prospecting 337); 0 closed rows | none |
| `sales_agent` → sales_teams.csv | none | 0 | none. 5 agents in sales_teams.csv never appear (Carl Lin, Carol Thompson, Elizabeth Anderson, Mei-Mei Johns, Natalya Ivanova). |

- Cosmetic only: the accounts table spells one sector `technolgy` (12 accounts, 1,058 closed deals). The spelling is consistent and the column is not used in any join, so it changes nothing. It is left as is.
- After the `GTXPro` fix, no feature column has a missing value. The in-fold imputation planned for F1 is therefore a guard (it should impute 0 rows, and the run will assert that).

### 1.3 `engage_year`: right-censoring or truncation?

- `engage_date` runs from 2016-10-20 to 2017-12-27. **`close_date` runs only from 2017-03-01 to 2017-12-31.**
- Lost deals close faster than won deals: median 14 vs 57 days. Duration is used only for this diagnosis, never as a feature.

| Engage period | Engaged | Closed (kept) | Open share dropped | Win rate [95% CI] |
|---|---|---|---|---|
| 2016 Q4 | 358 | 305 | 14.8% | **0.820** [0.773, 0.859] |
| 2017 Q1 | 1,619 | 1,322 | 18.3% | 0.679 [0.654, 0.704] |
| 2017 Q2 | 2,388 | 2,022 | 15.3% | 0.616 [0.595, 0.637] |
| 2017 Q3 | 2,770 | 2,012 | 27.4% | 0.600 [0.578, 0.621] |
| 2017 Q4 | 1,165 | 1,050 | 9.9% | 0.607 [0.577, 0.636] |

By month, the win rate is 0.81–0.82 for Nov–Dec 2016, then 0.73, 0.66 and 0.67 for Jan–Mar 2017, then 0.55–0.67. December 2017 rises again to 0.777 (n = 103). The monthly table is in the script output.

**Diagnosis.**
- **This is not classic right-censoring.** The share of open deals that gets dropped does not rise toward the end of the data. It peaks in July–August 2017 (32–34%) and is lowest in Q4 2017 (9.9%).
- **It is left truncation by the observation window.** A deal engaged before 2017-03-01 is in the data only if it closed after that date. Lost deals close fast, so the early cohort over-represents won deals: 972 deals, 72.8% won, against 61.5% for the rest. At the other edge, December 2017 deals appear only if they closed within days, which explains the December bump.
- **`engage_year` is just a flag for the 2016 cohort** (305 deals). Its univariate out-of-fold AUC is 0.518 on all rows and **exactly 0.500** on deals engaged on or after 2017-03-01. In deployment every new deal would carry a year the model has never seen.

**Recommendation: drop `engage_year`.** It costs nothing: CatBoost scores 0.5422 with it and 0.5408 without (fold std 0.016–0.018).

**`engage_month` mixes two effects.**
- It scores OOF AUC 0.565 on all rows but 0.546 after 2017-03-01, so part of it is the same window artifact (Nov–Feb and December).
- The rest is the within-quarter pattern in 1.4, which survives the window. The ladder therefore **replaces** `engage_month` with a within-quarter encoding.

### 1.4 Within-quarter timing: the one real pattern

Win rate by month of the quarter (n in brackets):

| Engage quarter | 1st month | 2nd month | 3rd month |
|---|---|---|---|
| 2017 Q2 | 0.567 (753) | 0.622 (588) | 0.665 (681) |
| 2017 Q3 | 0.554 (796) | 0.633 (539) | 0.628 (677) |
| 2017 Q4 | 0.569 (703) | 0.643 (244) | 0.777 (103) |
| 2017 Q1 (left-truncated) | 0.733 (247) | 0.660 (420) | 0.672 (655) |

Win rate by days to quarter end (7-day bins, n = 368–641 per bin, except 65 in the last):

| Days to quarter end | 0–6 | 7–13 | 14–41 | 42–55 | 56–69 | 70–91 |
|---|---|---|---|---|---|---|
| Win rate | 0.584 | 0.661 | **0.692–0.736** | 0.586–0.604 | 0.601–0.649 | **0.554–0.580** |

- The pattern is the same in all three unaffected quarters, and it holds after 2017-03-01: weeks-to-quarter-end scores OOF AUC 0.565 on all rows and 0.567 after.
- It is not monotone. The planned "last 2 weeks of the quarter" flag shows nothing (0.619 vs 0.634), so it is replaced by `days_to_quarter_end`.
- **Caveat:** this is engage-time information, so it is allowed. But the data is synthetic, and the pattern may come from the data generator rather than real sales behaviour.

### 1.5 Candidate signals

Method:
- Win rate with n and a 95% Wilson interval; levels with n < 30 are marked "too small". The full tables are in the script output.
- χ² test of whether the win rate differs across levels.
- OOF AUC: a smoothed target mean (m = 20) fit on the training folds and scored on the saved Phase 1 test folds.
- "Post-window" means test rows engaged on or after 2017-03-01.

| Candidate | Levels | χ² p | Win-rate range | Levels with n < 30 | OOF AUC: all / post-window | Verdict |
|---|---|---|---|---|---|---|
| `sales_agent` | 30 | 0.264 | 0.550–0.704 | 0 | 0.507 / 0.505 | no signal* |
| `manager` | 6 | 0.786 | 0.621–0.644 | 0 | 0.497 / 0.498 | none |
| `regional_office` | 3 | 0.604 | 0.626–0.639 | 0 | 0.502 / 0.503 | none |
| `product` (fixed) | 7 | 0.372 | 0.603–0.648 | 1 (GTK 500, n = 25: 0.600 [0.41, 0.77]) | 0.502 / 0.505 | none |
| `account` | 85 | **0.943** | 0.531–0.750 | 0 | 0.491 / 0.497 | none: accounts vary *less* than chance would produce |
| `sector` | 10 | 0.971 | 0.612–0.648 | 0 | 0.486 / 0.490 | none |
| `office_location` | 15 | 0.419 | 0.561–0.750 | 0 | 0.497 / 0.494 | none |
| agent × product | 178 | 0.217 | 0.467–0.812 (n ≥ 30) | **89 cells (1,541 rows)** | 0.501 / 0.498 | none |
| account × product | 525 | 0.754 | 0.452–0.732 (13 cells with n ≥ 30) | **512 cells (6,228 rows, 93%)** | n/a | none; too sparse |
| price bucket | 7 | 0.372 | same as product | 1 | same as product | each product has one list price, so this *is* the product |
| price / account revenue (quintiles) | 5 | 0.967 | 0.625–0.636 | 0 | n/a | none |
| revenue / employees / year_established (quintiles) | 5 each | 0.279 / 0.360 / 0.469 | 0.614–0.648 | 0 | n/a | none |
| subsidiary or not | 2 | 0.740 | 0.631–0.636 | 0 | n/a | none |
| `engage_year` | 2 | n/a | 0.623–0.820 | 0 | 0.518 / **0.500** | window artifact |
| `engage_month` (month of year) | 12 | n/a | 0.554–0.803 | 0 | **0.565** / 0.546 | real signal + artifact |
| month in quarter | 3 | n/a | 0.581 / 0.647 / 0.674 (n = 2,506 / 1,893 / 2,312) | 0 | 0.546 / **0.549** | real |
| weeks to quarter end | 14 | n/a | 0.554–0.736 | 0 | **0.565 / 0.567** | real, the strongest |

\* About 224 deals per agent give a binomial SD of about 0.032 per agent's win rate. Thirty such agents with identical true rates would spread across roughly 0.56–0.70, which is what is observed. An AUC below 0.5 for `account` and `sector` is the usual sign of a target encoding that fits noise.

**Conditional check.** Does the key matter once the calendar is held fixed? This sums χ² within strata.

| Key | Within month-in-quarter | Within engage month |
|---|---|---|
| product | p = 0.40 | p = 0.081 |
| sales_agent | p = 0.12 | p = 0.51 |
| account | p = 0.69 | p = 0.72 |
| manager, regional_office, sector | p ≥ 0.54 | p ≥ 0.40 |

Taken alone, product × month-in-quarter (p = 8e-7) and agent × month-in-quarter (p = 7e-5) look like interactions. Both disappear once the main calendar effect is held fixed, so they are that effect showing through.

### 1.6 History and recency preview

**Time-aware history.**
- History for a test row = training-fold deals with the same key that closed strictly before the row's `engage_date`, smoothed toward the training-fold rate with m = 20. Rows engaged before 2017-03-01 have no history (15%).

| History of | Win-rate AUC: all / post-window | Prior-deal count AUC | Rows with no history |
|---|---|---|---|
| sales_agent | 0.491 / 0.498 | 0.463 | 15.3% |
| account | 0.481 / 0.483 | 0.467 | 16.4% |
| product | 0.474 / 0.485 | 0.468 | 15.0% |

- The counts are below 0.5 only because they grow with calendar time, and later deals win less (the 1.3 artifact). They are a clock, not a measure of skill.
- Any history feature also marks the left-truncated cohort, because its rows are exactly the ones with no history. That is why every rung reports a **post-window AUC**.

**Label-free recency.**
- Built from the full engagement log, including open deals and using strictly earlier dates. No outcomes are involved.
- `days_since_account_prev_engagement`: AUC 0.537 on all rows, 0.517 post-window. Most of it is the early window, where accounts have few earlier engagements.
- `agent_engagements_prev_30d`: AUC 0.450 on all rows (0.550 if the sign is flipped), 0.474 post-window (0.527 flipped). It is weak and probably tracks monthly deal volume, which follows the calendar.

### 1.7 Account overlap between train and test folds

- In every saved fold, the test fold contains **all 85 accounts and all 30 agents**, and every one of them also appears in training (100% of test rows).
- Each account has 38–193 deals (median 75) and is served by a median of 10 agents.
- So the Phase 1 numbers measure performance on *known* accounts. GroupKFold by account (about 17 unseen accounts per test fold) measures cold-start accounts.
- Given account χ² p = 0.94, the two are expected to differ little.

Time-split options. Train = deals closed before the cutoff; test = deals engaged on or after it; deals still open at the cutoff are left out.

| Cutoff | Train | Test | Left out |
|---|---|---|---|
| 2017-07-01 | 2,679 (66.6% won) | 3,062 (60.2%) | 970 |
| **2017-08-01** | **3,306 (63.3%)** | **2,266 (61.9%)** | 1,139 |
| 2017-09-01 | 4,091 (62.1%) | 1,727 (61.5%) | 893 |

### 1.8 CatBoost preview on the saved folds

Default CatBoost, seed 42, the Phase 1 folds. This is a sanity preview, not a result.

| Feature set | ROC-AUC (mean ± std) | Post-window AUC | Brier |
|---|---|---|---|
| F0 v1 as-is | 0.5422 ± 0.0180 (= Phase 1 exactly) | 0.5193 | 0.2317 |
| F0 without `engage_year` | 0.5408 ± 0.0158 | 0.5214 | 0.2319 |
| F1 (name fix, no `engage_year`) | 0.5395 ± 0.0156 | 0.5203 | 0.2319 |
| F1 + month_in_quarter + days_to_quarter_end | 0.5680 ± 0.0198 | 0.5545 | 0.2305 |
| **same without `engage_month` (= planned F2)** | **0.5519 ± 0.0191** | **0.5542** | 0.2322 |
| calendar only (engage_month + the 2 new) | 0.5613 ± 0.0167 | 0.5419 | 0.2322 |
| F0 without any date feature | **0.4876** ± 0.0214 | 0.4907 | 0.2353 |

What the preview shows:
- Dropping `engage_year` and fixing the product name change nothing measurable.
- The within-quarter features add about 0.03 post-window AUC.
- `engage_month` adds 0.016 on all rows and nothing post-window. That extra 0.016 is the window artifact.
- Without dates, the 12 static features are slightly worse than useless.

### 1.9 What this means for the ladder

1. Put the calendar group first, because it is the only group with evidence. Each later rung then tests one null hypothesis cleanly.
2. Keep agent, account and product history as **negative controls**. They are cheap, and the README needs model-level evidence for "did not help".
3. Drop groups with no signal and no special reason to test them (section 2.2).
4. Report a **post-window AUC** on every rung, and only accept a gain if it also shows there. Otherwise any feature touching absolute time can score by exploiting the window edges.

---

## 2. Plan

### 2.1 Feature ladder (cumulative)

| Rung | Change from the previous rung | Features (num + cat) | Evidence | Expectation |
|---|---|---|---|---|
| **F0** | v1 as-is: 14 v1 features from the **v1 processed file** (bug included) | 14 (8 + 6) | Phase 1 | Reproduces Phase 1 **exactly**, every model and every fold |
| **F1** | Data fixes: `GTXPro → GTX Pro` (real `series` and `sales_price`); in-fold imputers as a guard; **drop `engage_year`** | 13 (7 + 6) | 1.2, 1.3 | ≈ F0 (preview 0.5395) |
| **F2** | Replace `engage_month` with `month_in_quarter` (1–3) and `days_to_quarter_end` (0–91) | 14 (8 + 6) | 1.4, 1.8 | The main gain, visible post-window (preview 0.552 / 0.554) |
| **F3** | + `sales_agent` (categorical), `agent_hist_win_rate`, `agent_hist_n` (log1p) | 17 (10 + 7) | χ² p = 0.26; OOF 0.507; history 0.491 | Null (negative control) |
| **F4** | + `account_hist_win_rate`, `account_hist_n`, `product_hist_win_rate`, `product_hist_n`, `days_since_account_prev_engagement`, `agent_engagements_prev_30d` | 23 (16 + 7) | History 0.47–0.48; recency 0.517 / 0.527 post-window | Null to tiny; watch post-window |

Extra rows, run on the primary folds. They are not eligible for freezing.
- **F4-TE**, the comparison arm: F4 with the three `*_hist_win_rate` columns replaced by sklearn `TargetEncoder` (fold-safe OOF target encoding) of `sales_agent`, `account` and `product`. The counts stay time-aware.
- **D1 = F2 + `engage_month`.** Measures how much of the score is the window artifact.
- **D2 = calendar only** (`month_in_quarter`, `days_to_quarter_end`). Shows how much the 12 static features add on top of the calendar.

Unchanged v1 columns in every rung: `year_established, revenue, employees, sales_price, log_revenue, log_employees` (numeric) and `sector, office_location, series, product, manager, regional_office` (categorical). `series` is a function of `product`; it is kept because it is a v1 column.

### 2.2 Groups dropped, and why

| Requested item | Decision | Reason |
|---|---|---|
| agent × product win rate | **dropped** | χ² p = 0.22, OOF AUC 0.501, half the cells have n < 30 (1,541 rows); F3/F4 already give trees `sales_agent` and `product` |
| account × product | **dropped** | 512 of 525 cells have n < 30 (93% of rows); χ² p = 0.75 |
| `sales_price / account revenue` | **dropped** | Quintile χ² p = 0.97 (0.625–0.636); trees can already form it from `sales_price` and `revenue` |
| engage quarter | **dropped** | After 2017-03-01 the quarters are flat (0.616 / 0.600 / 0.607); before that it only encodes the window artifact |
| quarter-end flag (last 2 weeks) | **replaced** by `days_to_quarter_end` + `month_in_quarter` | The flag shows nothing (0.619 vs 0.634); the real curve peaks 14–41 days before quarter end |
| `engage_year` | **dropped** (F1) | Window artifact; post-window AUC exactly 0.500 |
| `engage_month` | **replaced** (F2) | Its extra over the within-quarter features is the window artifact (1.8); D1 reports the size |
| agent and account/product history | **kept as negative controls** | No signal expected, but cheap, and needed to show "did not help" with model evidence |

### 2.3 Feature definitions

All features are computed per training fold by `src/features.py`. History features are never computed once on the full data.

- **`month_in_quarter`** = `(engage_month − 1) mod 3 + 1`. **`days_to_quarter_end`** = days from `engage_date` to the last day of its quarter. Both depend only on `engage_date`.
- **Time-aware history** (`TimeAwareHistory(key, m=20)`, sklearn-style):
  - `fit(X_train, y_train)` stores, for each key value, the training rows' `close_date` (sorted) and their cumulative wins, plus `prior = mean(y_train)`.
  - `transform(X)` uses only the row's key and `engage_date`. It sets n = number of stored deals with the same key and `close_date < engage_date` (strictly before), and w = wins among them. It outputs `hist_win_rate = (w + m·prior) / (n + m)` and `hist_n = log1p(n)`.
  - A deal never sees itself: its own `close_date` is after its `engage_date` (minimum duration 1 day). The code asserts this.
  - The history pool is the **training fold only**, for training and test rows alike, so test labels can never enter. Deployment would also see test-fold deals; this choice is the conservative one.
  - `prior` is one constant per fold, so it carries no row-level information. `m = 20` is fixed in advance (about 10% of a typical agent's deals) and is **not tuned**.
- **OOF target encoding** (comparison arm only): `TargetEncoder(target_type="binary", smooth="auto", cv=5, shuffle=True, random_state=42)`, fit inside the training fold. Training rows get cross-fitted values; test rows get values fit on the full training fold.
- **Recency (label-free)**, computed from the full `sales_pipeline.csv` engagement log, including open deals, using strictly earlier `engage_date`s:
  - `days_since_account_prev_engagement`. A NaN (the account's first engagement, 1.3%) is filled in-fold with the training-fold maximum.
  - `agent_engagements_prev_30d`: the agent's engagements in [t − 30 days, t).
  - No outcome and no close date is involved, and this is the same information a live system would have.
- **In-fold imputers.** Numeric columns use the median and categorical columns the most frequent value, both fit on the training fold. After the fix they should impute 0 rows; the run logs the count and stops if it is not 0 in F1–F4.

### 2.4 Leakage guards

1. **Forbidden columns** never reach a model, and `build_feature_set` asserts it. The list: `close_value`, `log_close_value`, `close_date`, `close_year`, `close_month`, `deal_duration_days`, `deal_stage`, `target`, and any column whose name contains `close` or `duration`. `close_date` is read only inside `TimeAwareHistory.fit`, for training rows.
2. **Invariance test** (runs before any model):
   - Scramble the labels and `close_date`s of the test rows, then recompute their features. They must be unchanged.
   - Scramble the training rows instead. The test rows' features must change; this shows the check is not vacuous.
3. **Shuffled labels** (section 2.8).
4. Every learned transform is fit inside the training fold: scaler, encoders, imputers, history tables and TE.

### 2.5 Models: the same on every rung

| Model | Settings |
|---|---|
| Dummy (floor) | `DummyClassifier("most_frequent")`, exactly as in Phase 1 |
| Constant forecast (Brier reference) | `DummyClassifier("prior")`. A reference row: AUC 0.5, Brier ≈ 0.2327 (= 0.233) |
| LogReg v1 | Same pipeline: `StandardScaler` + `OneHotEncoder(handle_unknown="ignore")` → `LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)` |
| DualPathMLP v1 | Same class, layers, optimiser, early stopping and inner split |
| Hybrid v1 | 0.4·LogReg + 0.6·MLP, threshold tuned on the inner split as in Phase 1 |
| CatBoost | Default, `random_seed=42`, `cat_features` = the rung's categorical columns |
| LightGBM | Default `LGBMClassifier(random_state=42, verbose=-1, deterministic=True, force_row_wise=True)`; categoricals as pandas `category` with the training fold's categories |
| TabPFN (ceiling) | `TabPFNClassifier(random_state=42, device="cpu")`, default settings, as in Phase 1 |

- The v1 helpers hardcode their column lists (`NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`). Phase 2 therefore uses thin wrappers that take the rung's column lists and otherwise call the unchanged v1 code: `DualPathMLP`, the same `LogisticRegression` settings, and the same scaler and encoder.
- Only the input widths change. **Proof:** at F0, the wrappers must reproduce `results/benchmark_v1.csv` exactly.
- `lightgbm==4.7.0` is already installed in `.venv`. Stage 3 adds it to `requirements.txt`.

### 2.6 Evaluations

| Role | Split | Rungs | Notes |
|---|---|---|---|
| **PRIMARY** | Saved Phase 1 folds (`results/benchmark_v1_folds.csv`), loaded via `src/benchmark.py`; the run stops if they differ | F0–F4, F4-TE, D1, D2 | All headline numbers |
| **SECONDARY** | `GroupKFold(n_splits=5)` by `account` (deterministic, no shuffle), saved to `results/phase2_group_folds.csv` | F0–F4 | About 17 unseen accounts per test fold; account history is cold-start (prior, n = 0) by design |
| **DIAGNOSTIC** | Time split at **2017-08-01**: train = closed before, test = engaged on or after; deals open at the cutoff are left out | F0–F4, D1 | One test set (2,266 rows), 95% bootstrap CI for its AUC. Test history is frozen at the cutoff. Caveat: December test deals appear only if they closed fast. |

Plus a **post-window AUC** column everywhere: AUC on test rows engaged on or after 2017-03-01 (85% of rows), which are free of the left-truncation artifact.

### 2.7 Metrics and statistics

- **Per fold:** Phase 1's `fold_metrics` (ROC-AUC, PR-AUC, F1 / precision / recall / accuracy at 0.5, Brier, share predicted "won") plus `roc_auc_post_window`.
- **Headline:** ROC-AUC, PR-AUC and Brier as mean ± std across folds, always shown next to the Dummy (0.500 / 0.632 / 0.368) and the constant-63.2% Brier reference (0.233).
- **Paired bootstrap of the frozen rung vs F0**, for CatBoost and for TabPFN:
  - Each replicate (2,000, seed 42) resamples rows with replacement inside each test fold, then computes the per-fold metric difference on the same rows and averages over folds.
  - It reports ΔROC-AUC, ΔPR-AUC and ΔBrier, each with a mean, a 95% percentile CI and the share of replicates above 0, on all rows and post-window.
  - It captures test-sample noise only, not training variance, and says so.
- **Hybrid vs MLP, per rung:** mean ± std of the per-fold ΔAUC and ΔBrier, the number of folds where the Hybrid wins, and the paired bootstrap at the frozen rung. Reported whatever the answer.

### 2.8 Sanity checks that stop the run

1. **The folds match** the saved file (`save_or_verify_folds`), and the row order of the v2 processed file matches the folds' `opportunity_id`s.
2. **The Dummy reproduces Phase 1 exactly** (every metric, every fold). Stronger: **F0 reproduces `results/benchmark_v1.csv` exactly for all six Phase 1 models.** TabPFN on CPU was byte-identical across runs in Phase 1. If it ever differs, the run stops and reports the difference rather than loosening the check silently.
3. **Shuffled labels on the frozen rung:**
   - Permute `y` inside each training fold (seed 42) and run the *whole* fold pipeline, including history and TE tables, on the shuffled labels. Score against the true test labels.
   - Models: LogReg, CatBoost, LightGBM and TabPFN.
   - The run stops if any model's mean AUC is more than 0.03 from 0.5. The null SD of the 5-fold mean AUC is about 0.007, so 0.03 is more than 4 SD.
4. The invariance test (2.4.2), the forbidden-column assertion (2.4.1) and the zero-rows-imputed check (2.3).
5. Each fold's positive rate is within 0.002 of the overall rate (the existing check).

### 2.9 Freeze rule: which rung goes to Phase 3

The score is the mean of CatBoost's and TabPFN's fold-mean ROC-AUC on the primary folds. Start from F1. Accept a higher rung, compared with the **currently accepted** rung, only if all three hold:

- (a) the score rises by at least 0.005 on all rows;
- (b) it also rises by at least 0.003 post-window;
- (c) for at least one of the two models, the paired-bootstrap 95% CI of ΔAUC excludes 0.

Otherwise keep the lower rung. On a tie, the lower rung wins (parsimony). D1, D2 and F4-TE can never be frozen; F4-TE cannot because it uses outcomes of deals that closed *after* the scored deal was engaged, which the hard rules do not allow. The frozen rung must also not fall below F0 under GroupKFold, or the README flags it.

Expected result: **F2.**

### 2.10 Stage 3 work items, in commit order

1. **`src/process_data.py`:**
   - Add a minimal, commented `fix_product_names` option that maps `GTXPro` to `GTX Pro` before the products join.
   - `python -m src.process_data` writes the **unchanged** v1 file `processed_sales_data.csv`, plus `processed_sales_data_v2.csv` with the fix.
   - v1 (`main.py`, `--check-v1`) and Phase 1 keep reading the v1 file, so they stay reproducible.
2. **`src/features.py`:**
   - Transformers: `CalendarFeatures`, `TimeAwareHistory`, `OOFTargetEncoding`, `RecencyFeatures`, `InFoldImputer`.
   - `build_feature_set(rung)`, which returns a fold-fittable transformer plus the numeric and categorical column lists.
   - The forbidden-column assertion and the invariance self-test.
3. **`src/run_phase2.py`:**
   - Reuses `src.benchmark` (folds, `fold_metrics`, `set_seeds`, the model classes), plus the column-aware v1 wrappers and LightGBM.
   - CLI: `--eval primary|group|time`, `--rungs`, `--models`, then `--report`. Each (eval, rung, model) is cached under `results/phase2_parts/` (gitignored), so no single command runs past about 60 minutes.
   - `--report` writes `results/phase2_feature_ladder.csv` (per fold), `phase2_summary.csv` (mean ± std, Hybrid-vs-MLP, bootstrap), `phase2_group_kfold.csv`, `phase2_time_split.csv` and `phase2_env.json`.
   - It also writes `results/figures/phase2_feature_ladder.png` (ROC-AUC on all rows and post-window per rung and model, plus Brier with the 0.233 line) and `results/figures/phase2_shap_best.png`. The SHAP plot uses CatBoost's out-of-fold SHAP on the frozen rung, because TabPFN SHAP is too slow.
   - `requirements.txt`: add `lightgbm==4.7.0`.
4. **Sanity checks** (2.8) built into the run.
5. **Run twice and diff:**
   - The second run repeats every cheap model on every eval, plus TabPFN on F0 and the frozen rung (primary).
   - Expected: byte-identical on CPU. TabPFN stays on CPU, so no MPS tolerance is needed; if a difference appears, it is reported with its size.
6. **`scripts/get_data.md`:** the Maven Analytics "CRM Sales Opportunities" link (to be verified online in Stage 3) and the unzip steps into `data/raw/`.
7. **Polish commit:**
   - Remove absolute home-folder paths: PHASE1_PLAN.md (1) and `notebook/archive/data_loading_and_cleaning.ipynb` (3).
   - Add an MIT `LICENSE` and replace "Proprietary" in the README.
   - Reword the v1 print "Hybrid achieves best balance — proves necessity of both" (`src/phase3_pipeline.py:481`). Output text only, no numbers change.
   - No "publication-ready" or "novel" found. The grep's "improves" hits are false positives.
8. **README "Feature engineering" section:**
   - One honest opening sentence on signal strength.
   - The ladder chart.
   - An F0 → frozen-rung table next to the Dummy and the 0.233 reference.
   - GroupKFold and time-split results.
   - Three bullets, including the groups that did not help.
   - `key_results`: the rung to freeze, the best AUC per model, and whether any model beats Brier 0.233. If the best AUC is below 0.58, say so.

### 2.11 Runtime budget (M2 CPU)

| Block | TabPFN fits | Estimate |
|---|---|---|
| Primary: 8 rows (F0–F4, F4-TE, D1, D2) × 5 folds | 40 × ~2–2.5 min | ~90 min, split into two commands |
| GroupKFold: F0–F4 × 5 folds | 25 | ~55 min |
| Time split: 6 rungs × 1 fit | 6 | ~15 min |
| Shuffled-label check, frozen rung | 5 | ~12 min |
| Cheap models (Dummy, LogReg, MLP, Hybrid, CatBoost, LightGBM) across all of the above | n/a | ~20 min |
| Second run for the diff (cheap models everywhere + TabPFN on F0 and the frozen rung) | 10 | ~40 min |

Total: about 4 hours, with no single command over about 60 minutes. Fallback if TabPFN takes more than 3 min per fold: run GroupKFold TabPFN on F0, F2 and the frozen rung only, and say so.

### 2.12 Risks and caveats to state in the README

- **The only signal may come from the data generator.** Within-quarter timing is allowed (it is known at engage time), but real-world transfer is unknown.
- **The ceiling is low.** A best AUC around 0.55–0.58 is the realistic expectation, and much of it would be calendar. Phase 3 tuning cannot invent signal.
- **The window artifact** inflates any feature that touches absolute time: `engage_year`, `engage_month` and the history counts. Hence the post-window column and rule (b) of the freeze rule.
- **The time split has its own selection effect** at the right edge (December deals appear only if they closed fast).
- **Hybrid vs MLP:** in Phase 1 the Hybrid did not beat the MLP (0.518 vs 0.519 AUC). This is reported as measured on every rung.

---

## 3. Decisions (made unattended)

| # | Decision | Why |
|---|---|---|
| D1 | Skip Step 0 parts a–e. Only re-ran a 50-row TabPFN smoke test. | Already done and committed (`1f2b172`, `d3bd0d1`). |
| D2 | Commit a read-only analysis script, `scripts/phase2_stage1_analysis.py`, with this plan. | Makes every number above reproducible. It writes nothing and changes no code path. |
| D3 | **Drop `engage_year`** (F1). | Left-truncation artifact; post-window AUC exactly 0.500; costs 0.001 AUC. |
| D4 | **Replace `engage_month`** with `month_in_quarter` + `days_to_quarter_end` (F2), and report D1 (= F2 + `engage_month`) as a diagnostic. | Its extra over the within-quarter encoding is the same window artifact. This lowers the all-rows headline (CatBoost preview 0.568 → 0.552) but keeps what survives post-window. |
| D5 | **Reorder the ladder:** calendar (F2) before agent (F3) and account/product history (F4). | The calendar is the only group with evidence. Putting it first lets each later rung test a null cleanly instead of hiding the calendar gain inside F4. |
| D6 | **Drop** agent × product, account × product, price / revenue, engage quarter and the last-2-weeks flag. | No signal (2.2), or too sparse. |
| D7 | **Keep** agent and account/product history as negative controls. | The README must say with evidence which groups did not help. |
| D8 | **Add** `agent_engagements_prev_30d` to F4, which was not in the brief. | Label-free, known at engage time, with a weak post-window signal (0.527 flipped). It is cheap to test. |
| D9 | The history pool is training-fold deals only, closed strictly before `engage_date`, shrunk with m = 20 toward the fold's constant rate; m is not tuned. | Fold-safe, with test labels impossible to leak. Tuning m is Phase 3 territory. |
| D10 | **OOF target encoding** is a comparison arm (F4-TE) and **cannot be frozen**. | It uses outcomes of deals that closed after the scored deal was engaged, which is outside the allowed information. The time-aware version is the "cleaner one". |
| D11 | Label-free recency features use the full engagement log (all stages and folds). | No outcomes or close dates are involved, and a live system has exactly this information. |
| D12 | Keep the v1 processed file byte-identical; write the fixed table to `processed_sales_data_v2.csv`. | Keeps `main.py`, `--check-v1` and the Phase 1 benchmark reproducible, while F1+ use the fix. |
| D13 | Make the v1 models column-aware with thin wrappers; no change to their architecture or settings. | Required to feed them new columns. F0's exact reproduction of Phase 1 shows the wrappers change nothing. |
| D14 | LightGBM: default settings, seed 42, `deterministic=True`, `force_row_wise=True`. | Execution flags only, for reproducible runs. |
| D15 | TabPFN on CPU only. | About 2 min per fold is fast enough, and CPU was byte-identical in Phase 1; MPS could break determinism. |
| D16 | Time-split cutoff 2017-08-01. | The most balanced split (3,306 train / 2,266 test). The test months cover every month-in-quarter position. |
| D17 | Post-window AUC on every rung, and the freeze rule requires gains to appear there too. | Guards against features that only exploit the observation window. |
| D18 | Freeze rule as in 2.9 (≥ 0.005 all rows, ≥ 0.003 post-window, bootstrap CI excludes 0, parsimony). | Decided before seeing any Phase 2 result, so it cannot be fitted to them. |
| D19 | The second run (for the diff) repeats TabPFN only on F0 and the frozen rung. | Saves about 2 hours; CPU TabPFN determinism was already shown in Phase 1. |
| D20 | SHAP from CatBoost's out-of-fold predictions on the frozen rung. | TabPFN SHAP is too slow for 6,711 rows. |
| D21 | Leave the `technolgy` sector typo. | It is consistent and unused in joins, so it does not change any model. Fixing it would only churn data. |
| D22 | No status file written at this stage. | Nothing is blocked, and no status-file path is defined in the repo. |

## 4. Needs a human

Nothing blocks Stage 3.

Worth a look in the morning: **D4.** Replacing `engage_month` makes the honest all-rows AUC *lower* than keeping it, roughly 0.552 vs 0.568 in the CatBoost preview. The D1 diagnostic row will show both numbers side by side, so the choice can be reversed later without a re-run.
