"""Phase 2, Stage 1: read-only data analysis behind PHASE2_PLAN.md.

Prints every table the plan quotes. Reads data/raw/*.csv and
results/benchmark_v1_folds.csv; writes nothing.

Usage (from the project root):
    python scripts/phase2_stage1_analysis.py                     # ~1 min
    python scripts/phase2_stage1_analysis.py --catboost-preview  # + CatBoost feature-set preview, ~3 min
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist
from scipy.stats import chi2_contingency
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FOLDS = ROOT / "results" / "benchmark_v1_folds.csv"
WINDOW_START = pd.Timestamp("2017-03-01")  # first close_date in the data
MIN_N = 30
SMOOTH_M = 20  # pseudo-counts for smoothed win rates

pd.set_option("display.width", 200)


# --------------------------------------------------------------------------- data


def load_raw() -> dict[str, pd.DataFrame]:
    return {
        "accounts": pd.read_csv(RAW / "accounts.csv"),
        "products": pd.read_csv(RAW / "products.csv"),
        "pipeline": pd.read_csv(RAW / "sales_pipeline.csv", parse_dates=["engage_date", "close_date"]),
        "teams": pd.read_csv(RAW / "sales_teams.csv"),
    }


def closed_deals(raw: dict[str, pd.DataFrame], fix_product: bool = True) -> pd.DataFrame:
    """Won/Lost deals in raw order (= processed-file order), joined like process_data.py."""
    c = raw["pipeline"][raw["pipeline"].deal_stage.isin(["Won", "Lost"])].copy()
    if fix_product:
        c["product"] = c["product"].replace({"GTXPro": "GTX Pro"})
    c["won"] = (c.deal_stage == "Won").astype(int)
    c = c.merge(raw["accounts"], on="account", how="left").merge(raw["products"], on="product", how="left")
    c = c.merge(raw["teams"], on="sales_agent", how="left").reset_index(drop=True)
    folds = pd.read_csv(FOLDS)
    if not (c.opportunity_id.to_numpy() == folds.opportunity_id.astype(str).to_numpy()).all():
        raise RuntimeError("Row order differs from the saved folds.")
    c["fold"] = folds.fold.to_numpy()
    ed = c.engage_date
    c["engage_year"] = ed.dt.year
    c["engage_month"] = ed.dt.month
    c["engage_q"] = ed.dt.quarter
    c["month_in_quarter"] = (ed.dt.month - 1) % 3 + 1
    c["days_to_quarter_end"] = ((ed + pd.offsets.QuarterEnd(0)) - ed).dt.days
    c["weeks_to_quarter_end"] = c.days_to_quarter_end // 7
    c["post_window"] = ed >= WINDOW_START
    return c


# --------------------------------------------------------------------------- stats


def wilson(k, n, z: float = 1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def win_table(df: pd.DataFrame, key: str) -> pd.DataFrame:
    g = df.groupby(key, observed=True).won.agg(n="size", k="sum").reset_index()
    g["win_rate"] = g.k / g.n
    g["ci_lo"], g["ci_hi"] = wilson(g.k, g.n)
    g["flag"] = np.where(g.n < MIN_N, "too small", "")
    return g.drop(columns="k").sort_values("win_rate", ascending=False)


def chi2_test(df: pd.DataFrame, key: str) -> tuple[float, int, float]:
    stat, p, dof, _ = chi2_contingency(pd.crosstab(df[key], df.won))
    return stat, dof, p


def stratified_chi2(df: pd.DataFrame, key: str, strata: str) -> tuple[float, int, float]:
    """Sum of within-stratum chi2 statistics: does `key` matter once `strata` is held fixed?"""
    total, dof = 0.0, 0
    for _, d in df.groupby(strata):
        tab = pd.crosstab(d[key], d.won)
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        s, _, k, _ = chi2_contingency(tab)
        total += s
        dof += k
    return total, dof, float(chi2_dist.sf(total, dof))


def oof_te_auc(df: pd.DataFrame, key: str, rows=None) -> tuple[float, float]:
    """Univariate signal: smoothed target mean fit on the training folds, AUC on the test fold."""
    aucs = []
    for k in range(5):
        tr, te = df[df.fold != k], df[df.fold == k]
        if rows is not None:
            te = te[rows(te)]
        g = tr.groupby(key).won.agg(["sum", "size"])
        prior = tr.won.mean()
        enc = (g["sum"] + SMOOTH_M * prior) / (g["size"] + SMOOTH_M)
        aucs.append(roc_auc_score(te.won, te[key].map(enc).fillna(prior).astype(float)))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1))


def time_aware_history(df: pd.DataFrame, key: str) -> tuple[np.ndarray, np.ndarray]:
    """For each test-fold row: smoothed win rate and count of TRAINING-fold deals with the same
    key that closed strictly before this row's engage_date."""
    rate, count = np.full(len(df), np.nan), np.zeros(len(df))
    for k in range(5):
        tr = df[df.fold != k]
        prior = tr.won.mean()
        hist = {kv: g.sort_values("close_date") for kv, g in tr.groupby(key)}
        for i in np.flatnonzero((df.fold == k).to_numpy()):
            h = hist.get(df.at[i, key])
            if h is None:
                n, w = 0, 0
            else:
                n = int(np.searchsorted(h.close_date.to_numpy(), df.at[i, "engage_date"].to_datetime64(), side="left"))
                w = int(h.won.to_numpy()[:n].sum())
            rate[i] = (w + SMOOTH_M * prior) / (n + SMOOTH_M)
            count[i] = n
    return rate, count


def fold_auc(df: pd.DataFrame, score: np.ndarray, rows=None) -> tuple[float, float]:
    aucs = []
    for k in range(5):
        m = (df.fold == k).to_numpy()
        if rows is not None:
            m = m & rows  # pandas copy-on-write arrays are read-only, so no in-place &=
        aucs.append(roc_auc_score(df.won[m], score[m]))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1))


def header(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


# --------------------------------------------------------------------------- sections


def section_joins(raw: dict[str, pd.DataFrame]) -> None:
    header("1. Join audit: names in sales_pipeline.csv with no match in the lookup tables")
    pipe = raw["pipeline"]
    closed = pipe.deal_stage.isin(["Won", "Lost"])
    print(pipe.deal_stage.value_counts().to_string())
    for col, ref in [("product", raw["products"]["product"]), ("account", raw["accounts"]["account"]),
                     ("sales_agent", raw["teams"]["sales_agent"])]:
        unmatched = sorted(set(pipe[col].dropna()) - set(ref))
        print(f"\n{col}: {pipe[col].nunique()} distinct in pipeline; NaN in all rows {pipe[col].isna().sum()}, "
              f"in closed rows {pipe.loc[closed, col].isna().sum()}")
        for name in unmatched:
            print(f"   UNMATCHED '{name}': {int((pipe[col] == name).sum())} rows, "
                  f"{int(((pipe[col] == name) & closed).sum())} closed")
        if not unmatched:
            print("   no unmatched names")
        print(f"   in lookup table but never used: {sorted(set(ref) - set(pipe[col].dropna()))}")
    print("\nsector values (note the consistent typo 'technolgy'):", sorted(raw["accounts"].sector.unique()))


def section_censoring(raw: dict[str, pd.DataFrame], c: pd.DataFrame) -> None:
    header("2. engage_year: censoring / truncation check")
    pipe = raw["pipeline"]
    print(f"engage_date {pipe.engage_date.min().date()} .. {pipe.engage_date.max().date()}; "
          f"close_date {pipe.close_date.min().date()} .. {pipe.close_date.max().date()}")
    print("\nDuration (days, close - engage) by outcome; diagnostic only, never a feature:")
    print((c.close_date - c.engage_date).dt.days.groupby(c.deal_stage).describe().round(1).to_string())
    p = pipe.dropna(subset=["engage_date"]).copy()
    for label, key in [("year", p.engage_date.dt.year), ("quarter", p.engage_date.dt.to_period("Q").astype(str)),
                       ("month", p.engage_date.dt.to_period("M").astype(str))]:
        rows = []
        for g, d in p.groupby(key):
            cl = d[d.deal_stage.isin(["Won", "Lost"])]
            k, n = int((cl.deal_stage == "Won").sum()), len(cl)
            lo, hi = wilson(k, n)
            rows.append({"engage_" + label: g, "engaged": len(d), "closed": n, "open_share_dropped": 1 - n / len(d),
                         "win_rate": k / n, "ci_lo": lo, "ci_hi": hi, "flag": "too small" if n < MIN_N else ""})
        print(f"\nBy engage {label}:\n" + pd.DataFrame(rows).round(3).to_string(index=False))
    pre = c[~c.post_window]
    print(f"\nEngaged before {WINDOW_START.date()} (first close date): n={len(pre)}, win {pre.won.mean():.3f}, "
          f"min duration {(pre.close_date - pre.engage_date).dt.days.min()} days; "
          f"engaged on/after: n={(c.post_window).sum()}, win {c[c.post_window].won.mean():.3f}")


def section_calendar(c: pd.DataFrame) -> None:
    header("2b. Within-quarter calendar pattern (engage-time information)")
    print(c.pivot_table(index=c.engage_date.dt.to_period("Q").astype(str), columns="month_in_quarter",
                        values="won", aggfunc=["mean", "size"]).round(3).to_string())
    print("\nWin rate by days to quarter end (7-day bins):")
    print(c.groupby(pd.cut(c.days_to_quarter_end, list(range(-1, 100, 7))), observed=True)
          .won.agg(["mean", "size"]).round(3).to_string())
    print("\nPooled over all quarters:")
    for key in ["month_in_quarter", "engage_month"]:
        t = win_table(c, key).sort_values(key)
        print(f"   {key}: " + ", ".join(f"{k}: {w:.3f} (n={n})" for k, w, n in zip(t[key], t.win_rate, t.n)))
    print("\nOld 'quarter-end flag' (last 14 days of the quarter):")
    print(c.groupby(c.days_to_quarter_end <= 13).won.agg(["mean", "size"]).round(3).to_string())


def section_profiles(c: pd.DataFrame) -> None:
    header("3. Candidate signals: win rate, n, 95% Wilson CI (n < 30 marked 'too small'), chi2 heterogeneity")
    d = c.copy()
    d["agent_x_product"] = d.sales_agent + " | " + d["product"]
    d["account_x_product"] = d.account + " | " + d["product"]
    d["price_bucket"] = d.sales_price.astype(int).astype(str) + " (" + d["product"] + ")"
    d["price_rel_revenue_q"] = pd.qcut(d.sales_price / (d.revenue * 1e6), 5, labels=[f"Q{i}" for i in range(1, 6)])
    for col in ["revenue", "employees", "year_established"]:
        d[f"{col}_q"] = pd.qcut(d[col], 5, labels=[f"Q{i}" for i in range(1, 6)])
    d["is_subsidiary"] = d.subsidiary_of.notna()
    for key in ["sales_agent", "manager", "regional_office", "product", "account", "sector", "office_location",
                "agent_x_product", "account_x_product", "price_bucket", "price_rel_revenue_q", "revenue_q",
                "employees_q", "year_established_q", "is_subsidiary"]:
        t = win_table(d, key)
        s, dof, p = chi2_test(d, key)
        small = t[t.n < MIN_N]
        print(f"\n--- {key}: {len(t)} levels; chi2={s:.1f}, dof={dof}, p={p:.3g}; "
              f"levels with n<30: {len(small)} ({small.n.sum()} rows)")
        print((t if len(t) <= 30 else pd.concat([t.head(5), t.tail(5)])).round(3).to_string(index=False))


def section_signal_strength(c: pd.DataFrame) -> None:
    header("3b. Univariate out-of-fold AUC on the saved Phase 1 folds (smoothed target mean, m=20)")
    post = lambda d: d.post_window  # noqa: E731
    rows = []
    for key in ["engage_year", "engage_q", "engage_month", "month_in_quarter", "weeks_to_quarter_end",
                "sales_agent", "account", "product", "manager", "regional_office", "sector", "office_location"]:
        mu, sd = oof_te_auc(c, key)
        mu_p, _ = oof_te_auc(c, key, post)
        rows.append({"feature": key, "levels": c[key].nunique(), "oof_auc": mu, "std": sd, "oof_auc_post_window": mu_p})
    d = c.assign(agent_x_product=c.sales_agent + "|" + c["product"])
    mu, sd = oof_te_auc(d, "agent_x_product")
    rows.append({"feature": "agent_x_product", "levels": d.agent_x_product.nunique(), "oof_auc": mu, "std": sd,
                 "oof_auc_post_window": oof_te_auc(d, "agent_x_product", post)[0]})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\nJoint key x month_in_quarter cells (mixes the key with the calendar main effect):")
    for key in ["product", "sales_agent"]:
        joint = c[key].astype(str) + "|" + c.month_in_quarter.astype(str)
        s_, dof_, p_ = chi2_test(c.assign(joint=joint), "joint")
        print(f"   {key} x month_in_quarter: chi2={s_:.1f} dof={dof_} p={p_:.3g}")

    print("\nDoes the key matter once the calendar is held fixed? (sum of within-stratum chi2)")
    for key in ["product", "sales_agent", "account", "manager", "regional_office", "sector"]:
        for strata in ["month_in_quarter", "engage_month"]:
            s, dof, p = stratified_chi2(c, key, strata)
            print(f"   {key:<16} within {strata:<17} chi2={s:7.1f} dof={dof:4d} p={p:.3g}")

    print("\nTime-aware history preview (training-fold deals closed strictly before engage_date):")
    post_rows = c.post_window.to_numpy()
    for key in ["sales_agent", "account", "product"]:
        rate, count = time_aware_history(c, key)
        r, rs = fold_auc(c, rate)
        n, ns = fold_auc(c, count)
        print(f"   {key:<12} win-rate AUC {r:.4f} ± {rs:.4f} (post-window {fold_auc(c, rate, post_rows)[0]:.4f}); "
              f"count AUC {n:.4f} ± {ns:.4f}; rows with no history {np.mean(count == 0):.1%}")


def section_recency(raw: dict[str, pd.DataFrame], c: pd.DataFrame) -> None:
    header("3c. Label-free recency features from the full engagement log (all stages, strictly earlier dates)")
    log = raw["pipeline"].dropna(subset=["engage_date"])
    post = c.post_window.to_numpy()

    def per_key(key):
        return {k: np.sort(g.engage_date.to_numpy()) for k, g in log.dropna(subset=[key]).groupby(key)}

    def days_since_prev(key):
        groups, out = per_key(key), []
        for kv, t in zip(c[key], c.engage_date.to_numpy()):
            arr = groups.get(kv, np.array([], dtype="datetime64[ns]"))
            j = np.searchsorted(arr, t, side="left")
            out.append((t - arr[j - 1]) / np.timedelta64(1, "D") if j > 0 else np.nan)
        return np.array(out)

    def prev_30d(key):
        groups = per_key(key)
        return np.array([np.searchsorted(groups[kv], t, "left") - np.searchsorted(groups[kv], t - np.timedelta64(30, "D"), "left")
                         for kv, t in zip(c[key], c.engage_date.to_numpy())], dtype=float)

    for name, v in [("days_since_account_prev_engagement", days_since_prev("account")),
                    ("agent_engagements_prev_30d", prev_30d("sales_agent"))]:
        filled = np.where(np.isnan(v), np.nanmax(v) + 1, v)
        q = pd.qcut(pd.Series(filled).rank(method="first"), 5, labels=False)
        print(f"   {name}: NaN {np.isnan(v).mean():.1%}; raw AUC {roc_auc_score(c.won, filled):.4f}, "
              f"post-window {roc_auc_score(c.won[post], filled[post]):.4f}; "
              f"win rate by quintile {c.won.groupby(q).mean().round(3).tolist()}")


def section_overlap(c: pd.DataFrame) -> None:
    header("4. Account / agent overlap between train and test folds; time-split sizes")
    for k in range(5):
        tr, te = c[c.fold != k], c[c.fold == k]
        parts = []
        for key in ["account", "sales_agent"]:
            unseen = te[key].nunique() - te.loc[te[key].isin(set(tr[key])), key].nunique()
            parts.append(f"{key}: {te[key].nunique()} in test, {unseen} unseen in train, "
                         f"{te[key].isin(set(tr[key])).mean():.0%} of test rows seen")
        print(f"   fold {k}: " + "; ".join(parts))
    print(f"   deals per account: median {c.account.value_counts().median():.0f} "
          f"[{c.account.value_counts().min()}, {c.account.value_counts().max()}]; "
          f"agents per account: median {c.groupby('account').sales_agent.nunique().median():.0f}")
    print("\nTime split (train = closed before cutoff, test = engaged on/after cutoff):")
    for cut in ["2017-07-01", "2017-08-01", "2017-09-01"]:
        tr, te = c[c.close_date < cut], c[c.engage_date >= cut]
        print(f"   {cut}: train {len(tr)} (win {tr.won.mean():.3f}), test {len(te)} (win {te.won.mean():.3f}), "
              f"left out (open at cutoff) {len(c) - len(tr) - len(te)}")


def section_catboost_preview(raw: dict[str, pd.DataFrame]) -> None:
    header("5. CatBoost (default settings, seed 42) preview of candidate feature sets on the saved folds")
    from catboost import CatBoostClassifier

    bug = closed_deals(raw, fix_product=False)
    fix = closed_deals(raw, fix_product=True)
    # v1's processed values for the GTXPro rows: series "Unknown", sales_price = all-row median.
    bug["series"] = bug["series"].fillna("Unknown")
    bug["sales_price"] = bug["sales_price"].fillna(bug["sales_price"].median())
    num = ["year_established", "revenue", "employees", "sales_price", "engage_year", "engage_month"]
    cat = ["sector", "office_location", "series", "product", "manager", "regional_office"]

    def frame(d):
        x = d[num + cat].copy()
        x.insert(6, "log_revenue", np.log1p(d.revenue))
        x.insert(7, "log_employees", np.log1p(d.employees))
        return x

    f0, f1 = frame(bug), frame(fix).drop(columns="engage_year")
    calendar = fix[["month_in_quarter", "days_to_quarter_end"]]
    configs = {
        "F0 v1 as-is": f0,
        "F0 - engage_year": f0.drop(columns="engage_year"),
        "F1 (name fix, no engage_year)": f1,
        "F1 + month_in_quarter + days_to_quarter_end": pd.concat([f1, calendar], axis=1),
        "  same, without engage_month": pd.concat([f1.drop(columns="engage_month"), calendar], axis=1),
        "calendar only (3 features)": pd.concat([f0[["engage_month"]], calendar], axis=1),
        "F0 without any date feature": f0.drop(columns=["engage_year", "engage_month"]),
    }
    y, fold, post = fix.won.to_numpy(), fix.fold.to_numpy(), fix.post_window.to_numpy()
    for name, X in configs.items():
        auc, auc_post, brier = [], [], []
        for k in range(5):
            tr, te = fold != k, fold == k
            model = CatBoostClassifier(random_seed=42, verbose=0, allow_writing_files=False,
                                       cat_features=[c for c in cat if c in X.columns])
            p = model.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            auc.append(roc_auc_score(y[te], p))
            auc_post.append(roc_auc_score(y[te][post[te]], p[post[te]]))
            brier.append(brier_score_loss(y[te], p))
        print(f"   {name:<46} ROC-AUC {np.mean(auc):.4f} ± {np.std(auc, ddof=1):.4f} | "
              f"post-window {np.mean(auc_post):.4f} | Brier {np.mean(brier):.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catboost-preview", action="store_true", help="also fit CatBoost on 7 feature sets (~3 min)")
    args = parser.parse_args()
    raw = load_raw()
    c = closed_deals(raw)
    print(f"{len(c):,} closed deals, win rate {c.won.mean():.4f}")
    section_joins(raw)
    section_censoring(raw, c)
    section_calendar(c)
    section_profiles(c)
    section_signal_strength(c)
    section_recency(raw, c)
    section_overlap(c)
    if args.catboost_preview:
        section_catboost_preview(raw)


if __name__ == "__main__":
    main()
