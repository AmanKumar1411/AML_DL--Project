"""Phase 1 honest benchmark.

Scores the v1 models (LogReg, DualPathMLP, 40/60 Hybrid) with leak-free,
stratified 5-fold cross-validation, between a floor (DummyClassifier) and a
ceiling (TabPFN), with CatBoost as a strong classical baseline.

The v1 code is reused unchanged from ``src.phase3_pipeline``. Every
preprocessing step is fit inside the training fold. Anything tuned (MLP early
stopping, the hybrid threshold) uses an inner validation split of the training
fold, never the test fold.

Usage (from the project root):
    python -m src.benchmark --out results/benchmark_v1.csv
    python -m src.benchmark --check-v1      # reproduce v1's single-split numbers
"""
from __future__ import annotations

import os

# TabPFN refuses more than 5,000 training rows on CPU unless this guard is lifted.
# It is an execution guard, not a model setting, and must be set before tabpfn is imported.
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")

import argparse
import json
import platform
import random
import time
from importlib import metadata
from pathlib import Path
from typing import Callable, Protocol

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.phase3_pipeline import (
    CATEGORICAL_FEATURES,
    FEATURES,
    PROJECT_ROOT,
    RESULTS_DIR,
    TARGET,
    DualPathMLP,
    build_ml_model,
    load_dataset,
    prepare_data,
    split_numeric_categorical,
)

SEED = 42
N_FOLDS = 5
INNER_VAL_SIZE = 0.2  # v1's inner validation split (see phase3_pipeline.fit_models)
HYBRID_ML_WEIGHT, HYBRID_DL_WEIGHT = 0.4, 0.6  # v1 blend (phase3_pipeline.run_hybrid_model)
V1_THRESHOLD_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)  # v1's grid (phase3_pipeline.run_threshold_analysis)

# Columns written per fold; "_at_05" means the 0.5 decision threshold.
METRICS = [
    "roc_auc",
    "pr_auc",
    "f1_at_05",
    "precision_at_05",
    "recall_at_05",
    "accuracy_at_05",
    "brier",
    "pred_pos_rate_at_05",  # diagnostic: share of test rows predicted "won"
    "tuned_threshold",  # hybrid only: threshold chosen on the inner validation split
    "f1_at_tuned",  # hybrid only: test F1 at that threshold
]

Folds = list[tuple[np.ndarray, np.ndarray]]


class ProbaClassifier(Protocol):
    """Anything with sklearn-style ``fit`` and ``predict_proba`` (n_rows x 2)."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ProbaClassifier": ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


# --------------------------------------------------------------------------- setup


def set_seeds(seed: int = SEED) -> None:
    """Seed every RNG the benchmark touches and ask torch for deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    # warn_only: an op without a deterministic kernel warns instead of crashing.
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_xy() -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Return v1's modelling table: the same rows (dropna) and the same 14 features."""
    df = load_dataset()
    model_df = df.dropna(subset=FEATURES + [TARGET])
    X = model_df[FEATURES].reset_index(drop=True)
    y = model_df[TARGET].astype(int).to_numpy()
    ids = model_df["opportunity_id"].reset_index(drop=True)
    return X, y, ids


def make_folds(y: np.ndarray) -> Folds:
    """Stratified 5-fold split shared by every model."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))


def save_or_verify_folds(folds: Folds, ids: pd.Series, path: Path) -> None:
    """Save the fold of every row; if a saved file exists, stop when it differs."""
    fold_of = np.empty(len(ids), dtype=np.int64)
    for k, (_, test_idx) in enumerate(folds):
        fold_of[test_idx] = k
    table = pd.DataFrame({"row_index": np.arange(len(ids)), "opportunity_id": ids.astype(str), "fold": fold_of})

    if path.exists():
        saved = pd.read_csv(path)
        same = (
            len(saved) == len(table)
            and (saved["row_index"].to_numpy() == table["row_index"].to_numpy()).all()
            and (saved["opportunity_id"].astype(str).to_numpy() == table["opportunity_id"].to_numpy()).all()
            and (saved["fold"].to_numpy() == table["fold"].to_numpy()).all()
        )
        if not same:
            raise RuntimeError(f"Folds differ from {path}. Delete it only if the data change is intended.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def inner_split(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """v1's inner train/validation split of a training fold.

    Same call as ``fit_models`` (test_size 0.2, stratified, seed 42); the
    permutation depends only on n, y and the seed, so the rows are identical.
    """
    return train_test_split(np.arange(len(y)), test_size=INNER_VAL_SIZE, random_state=SEED, stratify=y)


# --------------------------------------------------------------------------- models


def _two_columns(p: np.ndarray) -> np.ndarray:
    """Turn P(won) into sklearn's (n_rows x 2) predict_proba layout."""
    return np.column_stack([1.0 - p, p])


class DualPathMLPV1:
    """v1 DualPathMLP, trained exactly as ``phase3_pipeline.fit_models`` does."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "DualPathMLPV1":
        y = np.asarray(y)
        self.X_train_ = X
        # v1 scaler + one-hot encoder, fit on the whole training fold.
        X_num, _, X_cat, _, _ = split_numeric_categorical(X, X)
        tr, val = inner_split(y)
        self.model_ = DualPathMLP(num_input_dim=X_num.shape[1], cat_input_dim=X_cat.shape[1], random_state=SEED)
        # Early stopping watches the inner validation split, never the test fold.
        self.model_.fit(X_num[tr], X_cat[tr], y[tr], X_num[val], X_cat[val], y[val])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # v1's function refits the same (deterministic) scaler/encoder on the
        # training fold and transforms X with it, so it is reused unchanged.
        _, X_num, _, X_cat, _ = split_numeric_categorical(self.X_train_, X)
        return _two_columns(self.model_.predict_proba(X_num, X_cat))


class HybridV1:
    """v1 hybrid: 0.4 * LogReg + 0.6 * DualPathMLP, both trained on the training fold.

    v1 chose its threshold on the test set. Here the same grid is searched on
    the inner validation split of the training fold (``tuned_threshold_``).
    """

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "HybridV1":
        y = np.asarray(y)
        self.ml_ = build_ml_model().fit(X, y)
        self.dl_ = DualPathMLPV1().fit(X, y)
        self.tuned_threshold_ = self._tune_threshold(X, y)
        return self

    @staticmethod
    def _blend(ml_p: np.ndarray, dl_p: np.ndarray) -> np.ndarray:
        return HYBRID_ML_WEIGHT * ml_p + HYBRID_DL_WEIGHT * dl_p

    def _tune_threshold(self, X: pd.DataFrame, y: np.ndarray) -> float:
        tr, val = inner_split(y)
        # LogReg is refit without the validation rows so its validation scores are
        # out-of-sample. The MLP never trained on them (they only drove early stopping),
        # which makes the tuned threshold mildly optimistic but keeps the test fold clean.
        ml_inner = build_ml_model().fit(X.iloc[tr], y[tr])
        p_val = self._blend(ml_inner.predict_proba(X.iloc[val])[:, 1], self.dl_.predict_proba(X.iloc[val])[:, 1])
        f1s = [f1_score(y[val], (p_val >= t).astype(int), zero_division=0) for t in V1_THRESHOLD_GRID]
        return float(V1_THRESHOLD_GRID[int(np.argmax(f1s))])  # ties -> lowest, like v1's idxmax

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return _two_columns(self._blend(self.ml_.predict_proba(X)[:, 1], self.dl_.predict_proba(X)[:, 1]))


class TabPFNCeiling:
    """TabPFNClassifier with default settings; only the seed and the CPU device are pinned."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TabPFNCeiling":
        from tabpfn import TabPFNClassifier  # heavy import, only when used

        self.model_ = TabPFNClassifier(random_state=SEED, device="cpu")
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X)


class CatBoostBaseline:
    """Strong classical baseline: CatBoost with default settings (seed pinned)."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "CatBoostBaseline":
        from catboost import CatBoostClassifier

        self.model_ = CatBoostClassifier(
            random_seed=SEED, verbose=0, allow_writing_files=False, cat_features=CATEGORICAL_FEATURES
        )
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X)


DUMMY_LABEL = "Dummy (floor)"
MODELS: dict[str, tuple[str, Callable[[], ProbaClassifier]]] = {
    "dummy": (DUMMY_LABEL, lambda: DummyClassifier(strategy="most_frequent")),
    "logreg": ("LogReg v1", build_ml_model),
    "mlp": ("DualPathMLP v1", DualPathMLPV1),
    "hybrid": ("Hybrid v1", HybridV1),
    "catboost": ("CatBoost (classical baseline)", CatBoostBaseline),
    "tabpfn": ("TabPFN (ceiling)", TabPFNCeiling),
}
DEFAULT_MODELS = "dummy,logreg,mlp,hybrid,catboost,tabpfn"


# --------------------------------------------------------------------------- evaluation


def fold_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Threshold-free metrics on P(won), plus the usual ones at ``threshold``."""
    y_pred = (p >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "f1_at_05": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_at_05": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_05": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy_at_05": float(accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, p)),
        "pred_pos_rate_at_05": float(y_pred.mean()),
    }


def evaluate_model(
    model_factory: Callable[[], ProbaClassifier],
    X: pd.DataFrame,
    y: np.ndarray,
    folds: Folds,
    name: str = "model",
) -> pd.DataFrame:
    """Fit a fresh model on each training fold, score it on the test fold.

    Returns one row per fold. A model exposing ``tuned_threshold_`` also gets
    its test F1 at that threshold (``f1_at_tuned``).
    """
    rows = []
    for k, (train_idx, test_idx) in enumerate(folds):
        set_seeds()  # every fit starts from the same RNG state, whatever ran before
        start = time.perf_counter()
        model = model_factory().fit(X.iloc[train_idx], y[train_idx])
        p = np.asarray(model.predict_proba(X.iloc[test_idx]))[:, 1]
        seconds = time.perf_counter() - start

        y_test = y[test_idx]
        threshold = getattr(model, "tuned_threshold_", None)
        rows.append(
            {
                "model": name,
                "fold": k,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "test_pos_rate": float(y_test.mean()),
                **fold_metrics(y_test, p),
                "tuned_threshold": np.nan if threshold is None else threshold,
                "f1_at_tuned": np.nan
                if threshold is None
                else float(f1_score(y_test, (p >= threshold).astype(int), zero_division=0)),
            }
        )
        # Timing goes to the console only, so two runs produce identical CSVs.
        print(f"  {name:<28} fold {k}: ROC-AUC {rows[-1]['roc_auc']:.4f}  ({seconds:.1f}s)", flush=True)
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, decimals: int | None = 4) -> pd.DataFrame:
    """Mean and sample std (ddof=1) across folds, per model, in run order."""
    grouped = results.groupby("model", sort=False)[METRICS]
    mean, std = grouped.mean(), grouped.std(ddof=1)
    columns = {}
    for metric in METRICS:
        columns[f"{metric}_mean"] = mean[metric]
        columns[f"{metric}_std"] = std[metric]
    summary = pd.DataFrame(columns)
    return (summary if decimals is None else summary.round(decimals)).reset_index()


def check_sanity(results: pd.DataFrame, positive_rate: float) -> None:
    """Stop if the floor or the folds don't behave as the theory says they must."""
    if (results["test_pos_rate"] - positive_rate).abs().max() > 0.002:
        raise RuntimeError("A fold's positive rate is more than 0.002 away from the overall rate.")
    dummy = results[results["model"] == DUMMY_LABEL]
    if dummy.empty:
        return
    if not (dummy["roc_auc"] == 0.5).all():
        raise RuntimeError("Dummy ROC-AUC is not exactly 0.5.")
    if not np.allclose(dummy["pr_auc"], dummy["test_pos_rate"], rtol=0, atol=1e-9):
        raise RuntimeError("Dummy PR-AUC does not equal the fold's positive rate.")
    print("Sanity checks passed: Dummy ROC-AUC = 0.5 and PR-AUC = positive rate in every fold.")


def check_v1_reproduction() -> None:
    """Score the v1 wrappers on v1's own 80/20 split and compare with the saved v1 CSV."""
    X_train, X_test, y_train, y_test = prepare_data(load_dataset())
    saved = pd.read_csv(RESULTS_DIR / "hybrid_model_comparison.csv").set_index("Setup")
    for setup, factory in [("ML only", build_ml_model), ("DL only", DualPathMLPV1), ("Hybrid", HybridV1)]:
        set_seeds()
        p = factory().fit(X_train, y_train.to_numpy()).predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, p)
        f1 = f1_score(y_test, (p >= 0.5).astype(int))
        d_auc, d_f1 = auc - saved.loc[setup, "ROC-AUC"], f1 - saved.loc[setup, "F1"]
        print(f"  {setup:<8} ROC-AUC {auc:.10f} (diff {d_auc:+.1e})  F1 {f1:.10f} (diff {d_f1:+.1e})")
        if max(abs(d_auc), abs(d_f1)) > 1e-9:
            raise RuntimeError(f"v1 reproduction failed for {setup}.")
    print("v1 reproduction check passed.")


def _tabpfn_model_version() -> str | None:
    """The TabPFN checkpoint version the package loads by default (e.g. 'v3.5')."""
    try:
        from tabpfn.settings import settings
    except ImportError:
        return None
    return str(settings.tabpfn.model_version.value)


def write_env(path: Path, models: list[str]) -> None:
    """Record versions and settings next to the results (no timestamp, so runs stay identical)."""
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn", "shap", "torch", "tabpfn", "catboost"]
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "seed": SEED,
        "n_folds": N_FOLDS,
        "models": models,
        "tabpfn_device": "cpu",
        "tabpfn_model_version": _tabpfn_model_version() if "tabpfn" in models else None,
        "TABPFN_ALLOW_CPU_LARGE_DATASET": os.environ.get("TABPFN_ALLOW_CPU_LARGE_DATASET"),
        "packages": versions,
    }
    path.write_text(json.dumps(info, indent=2) + "\n")


# --------------------------------------------------------------------------- chart

# Colours by role (floor / v1 / classical baseline / ceiling), validated with the
# dataviz palette checker; ink and rules from the reference palette.
FLOOR_COLOR, V1_COLOR, BASELINE_COLOR, CEILING_COLOR = "#898781", "#2a78d6", "#1baf7a", "#eb6834"
INK, INK_SECONDARY, GRID, BASELINE, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _role_color(name: str) -> str:
    if name == DUMMY_LABEL:
        return FLOOR_COLOR
    if "(ceiling)" in name:
        return CEILING_COLOR
    if "(classical baseline)" in name:
        return BASELINE_COLOR
    return V1_COLOR


def plot_ladder(summary: pd.DataFrame, path: Path, positive_rate: float, n_rows: int) -> None:
    """Horizontal bars of ROC-AUC and PR-AUC (mean ± std), floor -> v1 -> baseline -> ceiling."""
    names = summary["model"].tolist()
    colors = [_role_color(name) for name in names]
    y_pos = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(names) + 1.9), sharey=True, facecolor=SURFACE)
    for ax, metric, title in zip(axes, ["roc_auc", "pr_auc"], ["ROC-AUC", "PR-AUC (average precision)"]):
        mean, std = summary[f"{metric}_mean"].to_numpy(), summary[f"{metric}_std"].to_numpy()
        ax.set_facecolor(SURFACE)
        ax.barh(y_pos, mean, height=0.5, color=colors, zorder=2)
        ax.errorbar(mean, y_pos, xerr=std, fmt="none", ecolor=INK_SECONDARY, elinewidth=1.2, capsize=3, zorder=3)

        # Dashed reference line at the floor (the Dummy's mean).
        floor = mean[names.index(DUMMY_LABEL)] if DUMMY_LABEL in names else None
        if floor is not None:
            ax.axvline(floor, color=INK_SECONDARY, linestyle="--", linewidth=1, zorder=4)
            ax.text(floor + 0.012, -0.75, f"floor {floor:.3f}", ha="left", va="bottom", fontsize=8, color=INK_SECONDARY)

        for yi, m, s in zip(y_pos, mean, std):
            ax.text(m + s + 0.015, yi, f"{m:.3f} ± {s:.3f}", va="center", fontsize=8.5, color=INK)

        ax.set_xlim(0, 1)
        ax.set_ylim(len(names) - 0.5, -1.1)  # first model on top, room for the floor label
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)

    axes[0].set_yticks(y_pos, names, color=INK)
    fig.suptitle("Phase 1 benchmark: v1 between the floor and the ceiling", x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(
        0.01,
        0.01,
        f"Stratified {N_FOLDS}-fold CV (seed {SEED}) on {n_rows:,} closed deals, {positive_rate:.1%} won. "
        "Bars: mean across folds; whiskers: ±1 std. Dashed line: Dummy (always predicts 'won').",
        fontsize=8,
        color=INK_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 1 honest benchmark (leak-free 5-fold CV).")
    parser.add_argument("--out", type=Path, default=Path("results/benchmark_v1.csv"), help="per-fold results CSV")
    parser.add_argument("--models", default=DEFAULT_MODELS, help=f"comma-separated subset of: {DEFAULT_MODELS}")
    parser.add_argument("--check-v1", action="store_true", help="only reproduce v1's single-split numbers, then exit")
    args = parser.parse_args(argv)

    if args.check_v1:
        check_v1_reproduction()
        return

    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = [key.strip() for key in args.models.split(",")]
    unknown = [key for key in keys if key not in MODELS]
    if unknown:
        parser.error(f"unknown model(s) {unknown}; choose from {DEFAULT_MODELS}")

    set_seeds()
    X, y, ids = load_xy()
    folds = make_folds(y)
    save_or_verify_folds(folds, ids, out.with_name(f"{out.stem}_folds.csv"))
    write_env(out.with_name(f"{out.stem}_env.json"), keys)
    print(f"{len(y):,} rows, positive rate {y.mean():.4f}, {N_FOLDS} folds")

    results = pd.concat(
        [evaluate_model(MODELS[key][1], X, y, folds, name=MODELS[key][0]) for key in keys], ignore_index=True
    )
    check_sanity(results, float(y.mean()))

    summary = summarize(results)
    results.to_csv(out, index=False)
    summary.to_csv(out.with_name(f"{out.stem}_summary.csv"), index=False)
    # The chart uses unrounded means so its labels match tables rounded from the per-fold CSV.
    plot_ladder(summarize(results, decimals=None), out.parent / "figures" / f"{out.stem}_ladder.png", float(y.mean()), len(y))
    print(summary[["model", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std", "f1_at_05_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
