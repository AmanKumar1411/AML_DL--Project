# Phase 2 Plan: Feature Engineering

Status: **Stage 1 and Stage 2 done; revised twice after independent review and approved (section 5). Stage 3 executed; results in the README's "Feature engineering" section.**
Branch: `overnight-upgrade`. Model architectures and hyperparameters stay unchanged (that is Phase 3).

Every number below can be reproduced with a read-only script, which writes nothing itself. Its full output is committed as **`results/phase2_stage1_tables.txt`**: every level of every win-rate table, with n, 95% CI and the "too small" flag, plus the CatBoost preview. Two runs of that command were byte-identical.

```bash
python scripts/phase2_stage1_analysis.py                     # sections 1-4, ~1 min
python scripts/phase2_stage1_analysis.py --catboost-preview  # + section 5, ~3 min
python scripts/phase2_stage1_analysis.py --full --catboost-preview > results/phase2_stage1_tables.txt
```

Row subsets used throughout:

| Name | Rows | Definition |
|---|---|---|
| **all** | 6,711 | every closed deal |
| **post-window** | 5,739 | engaged on or after 2017-03-01, the first close date in the data (left edge trimmed) |
| **trimmed** | 5,636 | engaged 2017-03-01 to 2017-11-30 (both edges trimmed: also drops December 2017, whose win-rate bump is unexplained, see 1.3) |

## 0. Summary

- **Step 0 was already done** (commits `1f2b172`, `d3bd0d1`). TabPFN-3.5 is the Phase 1 ceiling: ROC-AUC 0.567 ± 0.010, Brier 0.229. CatBoost is kept as a classical baseline. Tonight's 50-row smoke test still loads and fits on CPU (4.9 s).
- **The static attributes carry no detectable signal.** Agent, manager, office, product, account, sector, price and every interaction tested match chance. χ² p is between 0.22 and 0.97 across 15 tables. Out-of-fold AUC is 0.49–0.51.
- **The only signal is when the deal was engaged, specifically its position within the quarter.**
  - Deals engaged 14–41 days before quarter end win 69–74%. Deals engaged in the quarter's first month win 55–57% (2017 Q2–Q4).
  - A target encoding of weeks-to-quarter-end alone scores OOF AUC 0.565. That roughly equals TabPFN's Phase 1 ceiling with all 14 features (0.567).
  - It survives trimming both edges of the data window: 0.567 post-window, **0.564 trimmed**. In the trimmed rows, months 1 / 2 / 3 of the quarter win 0.563 / 0.630 / 0.655.
  - The share of open deals dropped does not line up with the pattern (1.4). For example, months 1 and 3 of Q2 drop the same share (13.8% vs 14.0%) yet win 0.567 vs 0.665.
  - With every date feature removed, CatBoost drops to 0.488.
- **`engage_year` is a data-window artifact. Recommendation: drop it.**
  - The data only contains deals closed on or after 2017-03-01. So a deal engaged in late 2016 appears only if it was still open in March, which favours won deals.
  - The 2016 cohort wins 82%, against 62% for 2017.
  - On deals engaged after 2017-03-01, `engage_year` scores exactly AUC 0.500.
  - `engage_month` carries the same artifact on top of the real within-quarter signal. The plan replaces it with a within-quarter encoding.
- **December 2017 is unexplained.** It wins 77.7% (n = 103) although its deals closed fast, and fast closers usually win less. It may come from the data generator. December sits at exactly the values F2's new features describe (month 3, 0–30 days to quarter end), so every rung also reports an AUC with both edges trimmed.
- **History features will most likely not help.**
  - Time-aware agent, account and product win rates score AUC 0.47–0.49.
  - Their deal counts score 0.46–0.47, only because a count is a proxy for time.
  - They stay in the ladder as clearly labelled negative controls, so the README can say with evidence that they did not help.
- **Account overlap is total:** all 85 accounts appear in every train and test fold. GroupKFold by account is therefore the important secondary check.
- **Expected outcome: the best AUC will probably stay below 0.58.** A CatBoost preview of the planned F2 rung scores 0.552 on all rows, 0.554 post-window and 0.552 trimmed (F0: 0.542 / 0.519 / 0.515). If that holds, the README will say so plainly.
- **Runtime:** 56 TabPFN fits, about 2.2–2.7 h, within the 3 h budget. The 2 min/fold figure came from 14 features, so a timing gate first times one 23-feature F4 fold and re-estimates. If the new total is over 3 h, cuts decided in advance apply (2.11).

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

By month, the win rate is 0.81–0.82 for Nov–Dec 2016, then 0.73, 0.66 and 0.67 for Jan–Mar 2017, then 0.55–0.67. December 2017 rises again to 0.777 (n = 103). The monthly table is in `results/phase2_stage1_tables.txt`.

**Diagnosis.**
- **This is not classic right-censoring.** The share of open deals that gets dropped does not rise toward the end of the data. It peaks in July–August 2017 (32–34%) and is lowest in Q4 2017 (9.9%).
- **It is left truncation by the observation window.** A deal engaged before 2017-03-01 is in the data only if it closed after that date. Lost deals close fast, so the early cohort over-represents won deals: 972 deals, 72.8% won, against 61.5% for the rest.
- **The December 2017 bump (0.777, n = 103) is unexplained.** The first draft of this plan blamed the right edge, but the plan's own data contradicts that:
  - December deals did close fast (median 9 days for won deals, 14 for lost) because the data ends on 2017-12-31.
  - But fast closers usually win *less*: post-window deals closed within 30 days win 0.570 (n = 3,100).
  - December dropped only 14.9% of its deals as still open, so there is little sign that deals are missing.
  - It may come from the data generator. Because December is month 3 of its quarter and 0–30 days from quarter end, which are exactly F2's new features, every rung also reports a **trimmed** AUC without it (2.6).
- **`engage_year` is just a flag for the 2016 cohort** (305 deals). Its univariate out-of-fold AUC is 0.518 on all rows and **exactly 0.500** on deals engaged on or after 2017-03-01. In deployment every new deal would carry a year the model has never seen.

**Recommendation: drop `engage_year`.** It costs nothing: CatBoost scores 0.5422 with it and 0.5408 without (fold std 0.016–0.018).

**`engage_month` mixes two effects.**
- It scores OOF AUC 0.565 on all rows, but 0.546 post-window and 0.542 trimmed. So part of it is the left-edge artifact (Nov–Feb), plus the unexplained December bump.
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

- Months 1 → 3 rise in each of 2017 Q2–Q4 (in Q3, months 2 and 3 are level).
- It holds after trimming either edge. Weeks-to-quarter-end scores OOF AUC 0.565 on all rows, 0.567 post-window and 0.564 trimmed. In the trimmed rows, months 1 / 2 / 3 win 0.563 / 0.630 / 0.655 (n = 2,252 / 1,371 / 2,013).
- **Could dropping open deals create the gradient?** Won deals take longer, so a month that drops many still-open deals might look worse. The dropped share does not follow the gradient:

| Quarter | Dropped open share, months 1 / 2 / 3 | Win rate of closed deals, months 1 / 2 / 3 |
|---|---|---|
| 2017 Q2 | 13.8% / 18.6% / 14.0% | 0.567 / 0.622 / 0.665 |
| 2017 Q3 | **33.6% / 32.0%** / 13.1% | 0.554 / 0.633 / 0.628 |
| 2017 Q4 | **7.6%** / 13.8% / 14.9% | 0.569 / 0.643 / 0.777 |

What the dropped-share table shows:
- In Q2, months 1 and 3 drop the same share but differ by 0.10 in win rate.
- In Q4, month 1 drops the fewest deals of any 2017 month and still wins least in its quarter.
- In Q3, months 1 and 2 drop similar shares but differ by 0.08.
- Selection cannot be fully excluded. If every dropped deal had been won, or lost, the bounds overlap (Q2 month 1: 0.489–0.627; month 3: 0.572–0.712; the full table is in the tables file). But nothing in the data points to it.

Other properties of the pattern:
- It is not monotone. The planned "last 2 weeks of the quarter" flag shows nothing (0.619 vs 0.634), so it is replaced by `days_to_quarter_end`.
- **Caveat:** this is engage-time information, so it is allowed. But the data is synthetic, and the pattern may come from the data generator rather than real sales behaviour.

### 1.5 Candidate signals

Method:
- Win rate with n and a 95% Wilson interval; levels with n < 30 are marked "too small". **Every level of every table is in `results/phase2_stage1_tables.txt`**; this section shows the summary.
- χ² test of whether the win rate differs across levels.
- OOF AUC: a smoothed target mean (m = 20) fit on the training folds and scored on the saved Phase 1 test folds, for all, post-window and trimmed test rows.

| Candidate | Levels | χ² p | Win-rate range | Levels with n < 30 | OOF AUC: all / post-window / trimmed | Verdict |
|---|---|---|---|---|---|---|
| `sales_agent` | 30 | 0.264 | 0.550–0.704 | 0 | 0.507 / 0.505 / 0.504 | no signal* |
| `manager` | 6 | 0.786 | 0.621–0.644 | 0 | 0.497 / 0.498 / 0.496 | none |
| `regional_office` | 3 | 0.604 | 0.626–0.639 | 0 | 0.502 / 0.503 / 0.503 | none |
| `product` (fixed) | 7 | 0.372 | 0.603–0.648 | 1 (GTK 500, n = 25: 0.600 [0.41, 0.77]) | 0.502 / 0.505 / 0.504 | none |
| `account` | 85 | **0.943** | 0.531–0.750 | 0 | 0.491 / 0.497 / 0.498 | none: accounts vary *less* than chance would produce |
| `sector` | 10 | 0.971 | 0.612–0.648 | 0 | 0.486 / 0.490 / 0.489 | none |
| `office_location` | 15 | 0.419 | 0.561–0.750 | 0 | 0.497 / 0.494 / 0.495 | none |
| agent × product | 178 | 0.217 | 0.467–0.812 (n ≥ 30) | **89 cells (1,541 rows)** | 0.501 / 0.498 / 0.497 | none |
| account × product | 525 | 0.754 | 0.452–0.732 (13 cells with n ≥ 30) | **512 cells (6,228 rows, 93%)** | n/a | none; too sparse |
| price bucket | 7 | 0.372 | same as product | 1 | same as product | each product has one list price, so this *is* the product |
| price / account revenue (quintiles) | 5 | 0.967 | 0.625–0.636 | 0 | n/a | none |
| revenue / employees / year_established (quintiles) | 5 each | 0.279 / 0.360 / 0.469 | 0.614–0.648 | 0 | n/a | none |
| subsidiary or not | 2 | 0.740 | 0.631–0.636 | 0 | n/a | none |
| `engage_year` | 2 | n/a | 0.623–0.820 | 0 | 0.518 / **0.500 / 0.500** | window artifact |
| `engage_month` (month of year) | 12 | n/a | 0.554–0.803 | 0 | **0.565** / 0.546 / 0.542 | real signal + artifact |
| month in quarter | 3 | n/a | 0.581 / 0.647 / 0.674 (n = 2,506 / 1,893 / 2,312) | 0 | 0.546 / **0.549** / 0.546 | real |
| weeks to quarter end | 14 | n/a | 0.554–0.736 | 0 | **0.565 / 0.567 / 0.564** | real, the strongest |

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

| History of | Win-rate AUC: all / post-window / trimmed | Prior-deal count AUC: all / trimmed | Rows with no history |
|---|---|---|---|
| sales_agent | 0.491 / 0.498 / 0.500 | 0.463 / 0.487 | 15.3% |
| account | 0.481 / 0.483 / 0.485 | 0.467 / 0.491 | 16.4% |
| product | 0.474 / 0.485 / 0.488 | 0.468 / 0.492 | 15.0% |

- The counts are below 0.5 only because they grow with calendar time, and later deals win less (the 1.3 artifact). They are a clock, not a measure of skill, and they move toward 0.5 once the edges are trimmed.
- Any history feature also marks the left-truncated cohort, because its rows are exactly the ones with no history. That is why every rung reports post-window and trimmed AUCs.

**Label-free recency.**
- Built from the full engagement log, including open deals and using strictly earlier dates. No outcomes are involved.
- `days_since_account_prev_engagement`: AUC 0.537 on all rows, 0.517 post-window, 0.515 trimmed. Most of it is the early window, where accounts have few earlier engagements.
- `agent_engagements_prev_30d`: AUC 0.450 on all rows (0.550 if the sign is flipped), 0.474 post-window (0.527 flipped), 0.479 trimmed (0.521 flipped). It is weak and probably tracks monthly deal volume, which follows the calendar.

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

| Feature set | ROC-AUC (mean ± std) | Post-window AUC | Trimmed AUC | Brier |
|---|---|---|---|---|
| F0 v1 as-is | 0.5422 ± 0.0180 (= Phase 1 exactly) | 0.5193 | 0.5151 | 0.2317 |
| F0 without `engage_year` | 0.5408 ± 0.0158 | 0.5214 | 0.5160 | 0.2319 |
| F1 (name fix, no `engage_year`) | 0.5395 ± 0.0156 | 0.5203 | 0.5148 | 0.2319 |
| F1 + month_in_quarter + days_to_quarter_end | 0.5680 ± 0.0198 | 0.5545 | 0.5503 | 0.2305 |
| **same without `engage_month` (= planned F2)** | **0.5519 ± 0.0191** | **0.5542** | **0.5519** | 0.2322 |
| calendar only (engage_month + the 2 new) | 0.5613 ± 0.0167 | 0.5419 | 0.5375 | 0.2322 |
| F0 without any date feature | **0.4876** ± 0.0214 | 0.4907 | 0.4923 | 0.2353 |

What the preview shows:
- Dropping `engage_year` and fixing the product name change nothing measurable.
- The within-quarter features add about 0.03–0.04 AUC on post-window and trimmed rows (F1 → F2: 0.520 → 0.554 and 0.515 → 0.552). So the gain does not rest on December.
- `engage_month` adds 0.016 on all rows and nothing on post-window or trimmed rows. That extra 0.016 is the window artifact.
- Without dates, the 12 static features are slightly worse than useless.

### 1.9 What this means for the ladder

1. Put the calendar group first, because it is the only group with evidence. Each later rung then tests one null hypothesis cleanly.
2. Keep agent, account and product history as **negative controls**. They are cheap, and the README needs model-level evidence for "did not help".
3. Drop groups with no signal and no special reason to test them (section 2.2).
4. Report **post-window and trimmed AUCs** on every rung, and only accept a gain if it also shows post-window. Otherwise any feature touching absolute time can score by exploiting the window edges. The trimmed column also checks that nothing rests on the unexplained December bump.

---

## 2. Plan

### 2.1 Feature ladder (cumulative)

| Rung | Change from the previous rung | Features (num + cat) | Evidence | Expectation |
|---|---|---|---|---|
| **F0** | v1 as-is: 14 v1 features from the **v1 processed file** (bug included) | 14 (8 + 6) | Phase 1 | Reproduces Phase 1 **exactly**, every model and every fold |
| **F1** | Data fixes: `GTXPro → GTX Pro` (real `series` and `sales_price`); in-fold imputers as a guard; **drop `engage_year`** | 13 (7 + 6) | 1.2, 1.3 | ≈ F0 (preview 0.5395) |
| **F2** | Replace `engage_month` with `month_in_quarter` (1–3) and `days_to_quarter_end` (0–91) | 14 (8 + 6) | 1.4, 1.8 | The main gain, visible post-window and trimmed (preview 0.552 / 0.554 / 0.552) |
| **F3** | + `sales_agent` (categorical), `agent_hist_win_rate`, `agent_hist_n` (log1p) | 17 (10 + 7) | χ² p = 0.26; OOF 0.507; history 0.491 | Null (negative control) |
| **F4** | + `account_hist_win_rate`, `account_hist_n`, `product_hist_win_rate`, `product_hist_n`, `days_since_account_prev_engagement`, `agent_engagements_prev_30d` | 23 (16 + 7) | History 0.47–0.48; recency 0.517 / 0.527 post-window | Null to tiny; watch post-window |

Extra rows, run on the primary folds. They are not eligible for freezing.
- **F4-TE**, the comparison arm: F4 with the three `*_hist_win_rate` columns replaced by sklearn `TargetEncoder` (fold-safe OOF target encoding) of `sales_agent`, `account` and `product`. The counts stay time-aware. Run without TabPFN (2.11).
- **D1 = F2 + `engage_month`.** Measures how much of the score is the window artifact. Run with TabPFN, because reversing D4 depends on it.
- **D2 = calendar only** (`month_in_quarter`, `days_to_quarter_end`). Shows how much the 12 static features add on top of the calendar. Run without TabPFN.

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
- **OOF target encoding** (comparison arm only): `TargetEncoder(target_type="binary", smooth="auto", cv=5, shuffle=True, random_state=42)`, fit inside the training fold with **`fit_transform`**. Training rows get cross-fitted values (calling `fit` then `transform` would not cross-fit them); test rows get values fit on the full training fold.
- **Recency (label-free)**, computed from the full `sales_pipeline.csv` engagement log, including open deals, using strictly earlier `engage_date`s:
  - `days_since_account_prev_engagement`. A NaN (the account's first engagement, 1.3%) is filled **inside `RecencyFeatures` itself** with the training-fold maximum, before the generic imputers run.
  - `agent_engagements_prev_30d`: the agent's engagements in [t − 30 days, t).
  - No outcome and no close date is involved, and this is the same information a live system would have.
- **In-fold imputers.** Numeric columns use the median and categorical columns the most frequent value, both fit on the training fold. The v2 file does **not** apply v1's all-row fill of `series` and `sales_price` (`process_data.py:62-63`), so any gap reaches these imputers and the stop below can actually fire. The run logs how many values each imputer fills. After the fix they should fill none, so the run **stops if any of the 13 v1 columns of F1–F4 needs imputing**. New features fill their own documented gaps (above), so F4 cannot set off a false alarm.

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

| Role | Split | Rungs: models without TabPFN | Rungs: TabPFN | Notes |
|---|---|---|---|---|
| **PRIMARY** | Saved Phase 1 folds (`results/benchmark_v1_folds.csv`), loaded via `src/benchmark.py`; the run stops if they differ | F0–F4, F4-TE, D1, D2 | F0–F4, D1 | All headline numbers |
| **SECONDARY** | `GroupKFold(n_splits=5)` by `account` (deterministic, no shuffle), saved to `results/phase2_group_folds.csv` | F0–F4 | F0 and the frozen rung | About 17 unseen accounts per test fold; account history is cold-start (prior, n = 0) by design. The folds are not stratified, so each fold's positive rate is logged rather than checked against 0.002. |
| **DIAGNOSTIC** | Time split at **2017-08-01**: train = closed before, test = engaged on or after; deals open at the cutoff are left out | F0–F4, D1 | F0–F4, D1 | One test set (2,266 rows), 95% bootstrap CI for its AUC. Test history is frozen at the cutoff. The test set contains the unexplained December 2017 bump (103 rows), so the trimmed AUC (2,163 rows) is reported next to it. |

"Models without TabPFN" means Dummy, the constant forecast, LogReg v1, DualPathMLP v1, Hybrid v1, CatBoost and LightGBM. All of them together take seconds per fold.

Two extra AUC columns on every row of every evaluation:

| Column | Test rows scored | Purpose |
|---|---|---|
| `roc_auc_post_window` | engaged on or after 2017-03-01 (85% of rows) | free of the left-truncation artifact; used by freeze rule (b) |
| `roc_auc_trimmed` | engaged 2017-03-01 to 2017-11-30 (84% of rows) | also drops the unexplained December 2017 bump, which sits exactly on F2's new features. Reported, not used by the freeze rule. Any README claim that within-quarter timing is "the only real signal" must hold in this column too. |

The time split has no left-edge rows in its test set, so its post-window AUC equals its all-rows AUC; its trimmed AUC drops December.

### 2.7 Metrics and statistics

- **Per fold:** Phase 1's `fold_metrics` (ROC-AUC, PR-AUC, F1 / precision / recall / accuracy at 0.5, Brier, share predicted "won") plus `roc_auc_post_window` and `roc_auc_trimmed`.
- **Headline:** ROC-AUC, PR-AUC and Brier as mean ± std across folds, always shown next to the Dummy (0.500 / 0.632 / 0.368) and the constant-63.2% Brier reference (0.233).
- **Paired bootstrap.** One routine runs on the cached out-of-fold predictions, so it needs no refits:
  - Each replicate (2,000, seed 42) resamples rows with replacement inside each test fold, then computes the per-fold metric difference on the same rows and averages over folds.
  - It reports ΔROC-AUC, ΔPR-AUC and ΔBrier, each with a mean, a 95% percentile CI and the share of replicates above 0, on all, post-window and trimmed rows.
  - It captures test-sample noise only, not training variance, and says so.
  - It runs on these pairs, for CatBoost and for TabPFN:
    - **every step of the freeze rule** (2.9): each candidate rung vs the currently accepted one (F1→F2, then F2→F3 or F1→F3, and so on). This is what rule (c) uses.
    - **the frozen rung vs F0**: the headline comparison.
    - **F2 vs D1**: what keeping `engage_month` would add.
- **Hybrid vs MLP, per rung:** mean ± std of the per-fold ΔAUC and ΔBrier, the number of folds where the Hybrid wins, and the paired bootstrap at the frozen rung. Reported whatever the answer.

### 2.8 Sanity checks that stop the run

1. **The folds match** the saved file (`save_or_verify_folds`), and the row order of the v2 processed file matches the folds' `opportunity_id`s.
2. **The Dummy reproduces Phase 1 exactly** (every metric, every fold). Stronger: **F0 reproduces `results/benchmark_v1.csv` exactly for all six Phase 1 models.** TabPFN on CPU was byte-identical across runs in Phase 1. The five deterministic models must match exactly. For TabPFN, the run logs the largest difference in any per-fold metric. If it is 0, fine; if it is at most 1e-6, the run continues and adds a needs_human flag; above 1e-6 it stops. F0 passes the columns in v1's `FEATURES` order with the dtypes `pd.read_csv` gives them, because TabPFN's output depends on column order. F0's TabPFN result is therefore also a repeatability check against an independent earlier run.
3. **Shuffled labels:**
   - Permute `y` inside each training fold (seed 42) and run the *whole* fold pipeline, including history and TE tables, on the shuffled labels. Score against the true test labels.
   - On the **frozen rung**, with LogReg, CatBoost, LightGBM and TabPFN.
   - On **F4 and F4-TE**, with LogReg, CatBoost and LightGBM (about 1 min). The expected frozen rung, F2, has no transform that uses labels, so only these two rungs exercise `TimeAwareHistory` and the target encoding.
   - The run stops if any model's mean AUC is more than 0.03 from 0.5. The null SD of the 5-fold mean AUC is about 0.007, so 0.03 is more than 4 SD.
4. The invariance test (2.4.2), the forbidden-column assertion (2.4.1) and the zero-imputation check on the v1 columns (2.3).
5. **Positive rates.** On the primary folds, each fold's positive rate must be within 0.002 of the overall rate (the existing `check_sanity`). GroupKFold folds are not stratified, so there each fold's positive rate is only logged and written to `phase2_group_kfold.csv`.
6. **v1 untouched** (after the `process_data.py` edit, 2.10.1): the SHA-256 of `data/processed/processed_sales_data.csv` must equal its hash before the edit, and `python -m src.benchmark --check-v1` must pass.

### 2.9 Freeze rule: which rung goes to Phase 3

The score is the mean of CatBoost's and TabPFN's fold-mean ROC-AUC on the primary folds. Start from F1. Accept a higher rung, compared with the **currently accepted** rung, only if all three hold:

- (a) the score rises by at least 0.005 on all rows;
- (b) it also rises by at least 0.003 post-window;
- (c) for at least one of the two models, the 95% CI of ΔAUC (candidate vs currently accepted rung, all rows; 2.7) excludes 0.

Otherwise keep the lower rung. On a tie, the lower rung wins (parsimony). D1, D2 and F4-TE can never be frozen; F4-TE cannot because it uses outcomes of deals that closed *after* the scored deal was engaged, which the hard rules do not allow.

The trimmed AUC is reported for every step but is not part of the rule. The rule is unchanged from the first draft and was fixed before any Phase 2 run (the Stage 1 CatBoost preview was already known). The frozen rung must also not fall below F0 under GroupKFold, judged by CatBoost and LightGBM on every rung plus TabPFN on F0 and the frozen rung; otherwise the README flags it.

If the timing gate (2.11) removes TabPFN from a rung, that step of the rule uses CatBoost and LightGBM instead, and the README says so.

Edge case, decided in advance: the ladder is cumulative, so if F3 fails but F4 passes against the accepted rung, F4 is frozen as scored (only scored rungs can be frozen), and needs_human notes that the rejected agent features ride along.

Expected result: **F2.**

### 2.10 Stage 3 work items, in commit order

1. **`src/process_data.py`:**
   - Before editing, record the SHA-256 of `data/processed/processed_sales_data.csv` and copy the file aside, so a failed hash check can be undone without reverting code.
   - Add a minimal, commented `fix_product_names` option that maps `GTXPro` to `GTX Pro` before the products join. With the fix on, the all-row fill of `series` and `sales_price` is skipped (gaps are left to the in-fold imputers).
   - `python -m src.process_data` writes the **unchanged** v1 file `processed_sales_data.csv`, plus `processed_sales_data_v2.csv` with the fix.
   - v1 (`main.py`, `--check-v1`) and Phase 1 keep reading the v1 file, so they stay reproducible.
   - After regenerating, assert the v1 file's hash is unchanged and run `python -m src.benchmark --check-v1`. Stop if either fails.
2. **`src/features.py`:**
   - Transformers: `CalendarFeatures`, `TimeAwareHistory`, `OOFTargetEncoding`, `RecencyFeatures`, `InFoldImputer`.
   - `build_feature_set(rung)`, which returns a fold-fittable transformer plus the numeric and categorical column lists.
   - The forbidden-column assertion and the invariance self-test.
3. **`src/run_phase2.py`:**
   - Reuses `src.benchmark` (folds, `fold_metrics`, `set_seeds`, the model classes), plus the column-aware v1 wrappers and LightGBM.
   - CLI: `--eval primary|group|time|shuffled`, `--rungs`, `--models`, `--time-one-fold`, then `--report`. Each (eval, rung, model) is cached under `results/phase2_parts/` (gitignored), with its out-of-fold predictions for the bootstrap. No single command runs past about 60 minutes.
   - Execution order:
     1. the timing gate (2.11);
     2. primary, all models without TabPFN on all 8 rows;
     3. primary TabPFN on F0–F4 and D1, in two commands;
     4. the freeze decision;
     5. GroupKFold;
     6. time split;
     7. shuffled labels;
     8. `--report`.
   - `--report` writes `results/phase2_feature_ladder.csv` (per fold), `phase2_summary.csv` (mean ± std, Hybrid-vs-MLP, every bootstrap of 2.7, the freeze-rule steps), `phase2_group_kfold.csv` (with per-fold positive rates), `phase2_time_split.csv` and `phase2_env.json`.
   - It also writes `results/figures/phase2_feature_ladder.png` (ROC-AUC on all, post-window and trimmed rows per rung and model, plus Brier with the 0.233 line) and `results/figures/phase2_shap_best.png`. The SHAP plot uses CatBoost's out-of-fold SHAP on the frozen rung, because TabPFN SHAP is too slow.
   - `requirements.txt`: add `lightgbm==4.7.0`.
4. **Sanity checks** (2.8) built into the run.
5. **Run twice and diff:**
   - The second run repeats every model without TabPFN on every eval, plus **TabPFN on the frozen rung only** (primary). F0's TabPFN result is already compared byte for byte with Phase 1's independent run (2.8.2), so it already shows TabPFN repeats.
   - Expected: byte-identical on CPU. TabPFN stays on CPU, so no MPS tolerance is needed; if a difference appears, it is reported with its size.
6. **`scripts/get_data.md`:** the Maven Analytics "CRM Sales Opportunities" link (to be verified online in Stage 3) and the unzip steps into `data/raw/`.
7. **Polish commit:**
   - Remove absolute home-folder paths:
     - `notebook/archive/data_loading_and_cleaning.ipynb` (3).
     - PHASE1_PLAN.md (1, at `:135`). That file's header says the rest of the document is unchanged, so add a line to its deviation list noting the removed path.
   - Add an MIT `LICENSE` and replace "Proprietary" in the README.
   - Reword the unsupported claims in v1's printed text and comments. This changes output text only; no numbers change and `--check-v1` must still pass.
     - `src/phase3_pipeline.py:451`: the comment "hybrid improves decision confidence".
     - `:481`: the print "Hybrid achieves best balance — proves necessity of both".
     - `:692–693`: the prints "PHASE 3: HYBRID INNOVATION" and "fully reproducible".
     - Then `python -m py_compile src/phase3_pipeline.py`, because `--check-v1` never executes lines 481 or 692 (and `main.py` would rewrite committed results).
   - `PHASE1_PLAN.md:223`: "This proves the benchmark wraps v1 faithfully" is reworded, with the same deviation note.
   - No "publication-ready" or "novel" found. The notebooks' two "improves generalization" lines describe L2 regularisation in general and stay.
8. **README "Feature engineering" section:**
   - One honest opening sentence on signal strength.
   - The ladder chart.
   - An F0 → frozen-rung table next to the Dummy and the 0.233 reference.
     - It includes the **D1 row** (F2 + `engage_month`), so readers see why TabPFN's all-rows AUC may fall below Phase 1's 0.567.
     - It shows all-rows, post-window and trimmed AUC.
   - GroupKFold and time-split results, with a note that in F3/F4 every GroupKFold test row has `account_hist_n = 0`, a value that in training only marks the early, 72.8%-won cohort; a worse Brier there is a design effect, not evidence that history hurts.
   - One line on selection: the calendar features were chosen after scoring about 17 candidates on these same folds (and the time-split test rows were part of that analysis). The gain, about 0.035 post-window AUC against a null SD of about 0.007, is far above that selection noise.
   - Three bullets, including the groups that did not help.
   - Any claim that within-quarter timing is "the only real signal" must hold in the trimmed column as well.
   - `key_results`: the rung to freeze, the best AUC per model **over the eligible rungs F0–F4 only** (D1 reported separately), and whether any model beats Brier 0.233. If the best AUC is below 0.58, say so.

### 2.11 Runtime budget (M2 CPU)

The budget is about 3 hours. TabPFN dominates the runtime; every other model together takes seconds per fold (Phase 1: about 30 s for five models × 5 folds).

| Block | TabPFN fits | Estimate at 2–2.5 min per fit |
|---|---|---|
| Timing gate: F4 fold 0 (23 features), cached and reused as F4's fold 0 in the primary run | (0 extra) | ~3 min |
| Primary: TabPFN on F0–F4 and D1 × 5 folds (F4-TE and D2 without TabPFN) | 30 | 60–75 min, in two commands |
| GroupKFold: TabPFN on F0 and the frozen rung × 5 folds (models without TabPFN on F0–F4) | 10 | 20–25 min |
| Time split: F0–F4 and D1, 1 fit each | 6 | 12–15 min |
| Shuffled labels: TabPFN on the frozen rung × 5 folds (F4 and F4-TE without TabPFN) | 5 | 10–13 min |
| Second run for the diff: TabPFN on the frozen rung only (primary) | 5 | 10–13 min |
| Models without TabPFN, across all of the above and both runs; bootstrap; report | n/a | ~20 min |
| **Total** | **56** | **about 2.2–2.7 h** (56 × 2–2.5 min + 20 min) |

Compared with the first draft, this cuts 30 TabPFN fits:
- 10 from primary (F4-TE and D2);
- 15 from GroupKFold (F1–F4 except the frozen rung);
- 5 from the second run (F0).

**Timing gate.** Before the primary run, the gate fits TabPFN on fold 0 of **F2 (14 features) and F4 (23 features)**; both fits are cached and reused by the primary run. It projects the total by charging t(F2) to every planned fit on a rung with at most 15 features (F0, F1, F2, D1) and t(F4) to F3, F4 and any 17+ feature fit, plus 20 min, and prints it. If the total is above 3 h, the run applies these cuts in this order, stops cutting as soon as the projected total is 3 h or less, and the README lists every cut taken:
1. Drop TabPFN from the shuffled-label check. The leakage risk is in the feature pipeline, which LogReg, CatBoost and LightGBM already exercise. Saves 5 fits.
2. Time split: TabPFN on F0, the frozen rung and D1 only. Saves 3 fits.
3. Drop the second-run TabPFN fits, relying on F0's byte-for-byte match with Phase 1 to show repeatability. Saves 5 fits.
4. Primary: drop TabPFN on F3 and F4. CatBoost and LightGBM carry the negative-control evidence, and the freeze rule falls back to them for those steps (2.9). Saves 10 fits.
5. If still over 3 h (about 4.9 min per fold or slower): TabPFN only on primary F0, F2, D1 and the frozen rung; status "partial".

Any cut is recorded in needs_human as well as the README, since cuts 4 and 5 break the brief's "TabPFN on every feature rung".

No single command runs past about 60 minutes.

### 2.12 Risks and caveats to state in the README

- **The only signal may come from the data generator.** Within-quarter timing is allowed (it is known at engage time), but real-world transfer is unknown.
- **The ceiling is low.** A best AUC around 0.55–0.58 is the realistic expectation, and much of it would be calendar. Phase 3 tuning cannot invent signal.
- **The window artifact** inflates any feature that touches absolute time: `engage_year`, `engage_month` and the history counts. Hence the post-window column and rule (b) of the freeze rule.
- **December 2017's win-rate bump is unexplained.** It is 0.777 (n = 103), although its deals closed fast and fast closers usually win less, and few of its deals were dropped as open. It may come from the data generator. It sits exactly on F2's new features (month 3, 0–30 days to quarter end) and inside the time split's test set, so the trimmed column, which excludes it, is reported everywhere.
- **Selection by open-deal dropping** cannot be fully excluded for the within-quarter pattern, because the worst-case bounds overlap. The dropped share does not follow the pattern, though (1.4).
- **Hybrid vs MLP:** in Phase 1 the Hybrid did not beat the MLP (0.518 vs 0.519 AUC). This is reported as measured on every rung.

---

## 3. Decisions (made unattended)

| # | Decision | Why |
|---|---|---|
| D1 | Skip Step 0 parts a–e. Only re-ran a 50-row TabPFN smoke test. | Already done and committed (`1f2b172`, `d3bd0d1`). |
| D2 | Commit a read-only analysis script, `scripts/phase2_stage1_analysis.py`, and its full output, `results/phase2_stage1_tables.txt`. | Makes every number above reproducible and puts every per-level table on disk. The script writes nothing itself (the file is its redirected output) and changes no code path. |
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
| D17 | Post-window and trimmed AUCs on every rung of every evaluation. The freeze rule requires gains to appear post-window; the trimmed column is reported but not used by the rule. | The post-window column guards against features that only exploit the left edge. The trimmed column also drops the unexplained December bump, which sits on F2's new features. |
| D18 | Freeze rule as in 2.9 (≥ 0.005 all rows, ≥ 0.003 post-window, bootstrap CI excludes 0, parsimony). | Fixed before any Phase 2 run. (The Stage 1 CatBoost preview of F2 was already known, so "cannot be fitted to the results" would overstate it.) |
| D19 | The second run (for the diff) repeats TabPFN on the **frozen rung only**. | F0's TabPFN result is compared byte for byte with Phase 1's independent run, which already shows TabPFN repeats on CPU. |
| D20 | SHAP from CatBoost's out-of-fold predictions on the frozen rung. | TabPFN SHAP is too slow for 6,711 rows. |
| D21 | Leave the `technolgy` sector typo. | It is consistent and unused in joins, so it does not change any model. Fixing it would only churn data. |
| D22 | No status file written at this stage. | Nothing is blocked, and no status-file path is defined in the repo. |
| D23 | TabPFN runs only where its answer is needed: primary F0–F4 and D1; GroupKFold F0 and the frozen rung; time split F0–F4 and D1; shuffled labels on the frozen rung. A timing gate decides, before the main run, whether further cuts (2.11) apply. | Keeps the run within about 3 h (56 fits, about 2.2–2.7 h). CatBoost and LightGBM still cover every rung of every evaluation. |
| D24 | Recency features fill their own NaNs (training-fold maximum). The zero-imputation stop covers only the 13 v1 columns. | Prevents a false alarm on F4 during an unattended run. |
| D25 | Shuffled-label check also on F4 and F4-TE (LogReg, CatBoost, LightGBM). | F2 has no transform that uses labels, so only these rungs test `TimeAwareHistory` and the target encoding. |
| D26 | Paired bootstrap for every freeze step, frozen vs F0, and F2 vs D1, all on cached predictions. | Rule (c) needs the step-wise comparison; it costs no refits. |
| D27 | Hash the v1 processed file before and after the `process_data.py` edit, and run `--check-v1`. | Shows that D12 holds. |

## 4. Needs a human

Nothing blocks Stage 3.

Worth a look in the morning: **D4.** Replacing `engage_month` makes the honest all-rows AUC *lower* than keeping it, roughly 0.552 vs 0.568 in the CatBoost preview. The D1 diagnostic row will show both numbers side by side, so the choice can be reversed later without a re-run.

## 5. Review response (revision 2)

An independent review returned REVISE with two required changes and nine suggestions. All were adopted.

| # | Review point | Change | Where |
|---|---|---|---|
| R1 | **Required:** about 4 h is over the 3 h budget, and the fallback only triggers above 3 min per fold. | Cut to 56 TabPFN fits (about 2.2–2.7 h) exactly as proposed. Added a timing gate on a 23-feature F4 fold, with cuts decided in advance if the gate projects over 3 h. | 2.6, 2.11, D19, D23 |
| R2 | **Required:** the December explanation contradicts the plan's own data, and post-window only trims the left edge. | December bump now called unexplained, with the numbers that rule out the old explanation. Added `roc_auc_trimmed` (2017-03-01 to 2017-11-30) to every rung of every evaluation, the figure, the bootstrap and the README table, and recomputed the Stage 1 evidence on trimmed rows. The calendar signal holds: OOF 0.564; CatBoost F2 preview 0.552 vs F0 0.515. Freeze rule unchanged. | 0, 1.3, 1.4–1.9, 2.6, 2.7, 2.10.8, 2.12, D17 |
| S1 | Rule (c) needs a bootstrap for each step, not just frozen vs F0. | Bootstrap now runs for every freeze step, plus frozen vs F0 and F2 vs D1. | 2.7, 2.9, D26 |
| S2 | The shuffled-label check on F2 never tests history or target encoding. | Also run on F4 and F4-TE with LogReg, CatBoost and LightGBM. | 2.8.3, D25 |
| S3 | The zero-imputation stop would fire falsely on F4. | Recency features fill their own NaNs; the stop covers the v1 columns only. | 2.3, 2.8.4, D24 |
| S4 | GroupKFold folds are not stratified, so the 0.002 check would fail. | Positive rates are logged for GroupKFold; the check stays on the primary folds. | 2.6, 2.8.5 |
| S5 | Check that the v1 processed file is unchanged. | Hash before and after, plus `--check-v1`. | 2.8.6, 2.10.1, D27 |
| S6 | Could dropping open deals create the within-quarter gradient? | Dropped share by (quarter, month in quarter) with worst-case bounds, added to the script and 1.4. The dropped share does not follow the gradient, though selection cannot be fully excluded. | 1.4, 2.12 |
| S7 | The per-level Stage 1 tables are not on disk. | `results/phase2_stage1_tables.txt` committed (`--full`), byte-identical across two runs. | header, 1.5, D2 |
| S8 | Polish misses `phase3_pipeline.py:451` and `:692–693`, and PHASE1_PLAN.md's "unchanged" note. | Both added to the polish item, plus a deviation note in PHASE1_PLAN.md. | 2.10.7 |
| S9 | Show D1 in the README table, not only in the CSV. | The README's F0 → frozen table includes the D1 row and all three AUC columns. | 2.10.8 |

### Revision 3 (approved with optional suggestions, all adopted)

| # | Suggestion | Change |
|---|---|---|
| A1 | The v2 file would inherit v1's all-row fill, so the imputation stop could never fire. | v2 skips the fill (2.3, 2.10.1). |
| A2 | The timing gate charges 23-feature time to every fit; no rule beyond cut 4; cuts must reach needs_human. | Gate times F2 and F4 fold 0 and charges by width; cut 5 added; cuts go to needs_human (2.11). |
| A3 | Decide now what a tiny TabPFN mismatch on F0 does. | Exact for the five deterministic models; TabPFN ≤ 1e-6 continues with a flag, above stops; column order pinned (2.8.2). |
| A4 | State the selection caveat; D18 overstated. | README line added (2.10.8); D18 and 2.9 reworded. |
| A5 | Cumulative-ladder edge case. | Resolution written down (2.9). |
| A6 | GroupKFold `account_hist_n = 0` design effect. | README note (2.10.8). |
| A7 | key_results over F0–F4 only. | 2.10.8. |
| A8 | `TargetEncoder.fit_transform` for cross-fitting. | 2.3. |
| A9 | Back up the v1 processed file. | 2.10.1. |
| A10 | `PHASE1_PLAN.md:223` "proves"; `py_compile` after editing prints. | 2.10.7. |
