from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score


def race_metrics(df: pd.DataFrame, score_col: str) -> dict:
    rows = []
    for _, g in df.groupby("raceId"):
        g = g.sort_values(score_col, ascending=False).copy()
        g["pred_rank"] = np.arange(1, len(g) + 1)
        wr = int(g.loc[g.winner == 1, "pred_rank"].iloc[0])
        rho = spearmanr(g.pred_rank, g.finish_position, nan_policy="omit").statistic
        mae = float(np.mean(np.abs(g.pred_rank - g.finish_position)))
        nd = float(
            ndcg_score([g.relevance.to_numpy()], [g[score_col].to_numpy()], k=min(5, len(g)))
        )
        rows.append([wr == 1, wr <= 3, rho, mae, nd])
    a = np.asarray(rows, float)
    return {
        "n_races": len(a),
        "top1": a[:, 0].mean(),
        "winner_top3": a[:, 1].mean(),
        "spearman": np.nanmean(a[:, 2]),
        "rank_mae": a[:, 3].mean(),
        "ndcg5": a[:, 4].mean(),
    }


def run_uploaded_results_summary(holdout_csv: str, feature_experiments_csv: str) -> pd.DataFrame:
    """Reproduce the archived summary of the original notebook's measured results."""
    h = pd.read_csv(holdout_csv)
    fe = pd.read_csv(feature_experiments_csv)
    h["grid_score"] = -pd.to_numeric(h.grid_effective, errors="coerce").fillna(99)
    baseline = {"feature_set": "GRID_BASELINE_RECOMPUTED", **race_metrics(h, "grid_score")}
    cols = [
        c
        for c in [
            "feature_set",
            "n_races",
            "top1",
            "winner_top3",
            "spearman",
            "rank_mae",
            "ndcg5",
            "n_features",
        ]
        if c in fe
    ]
    return pd.concat([pd.DataFrame([baseline]), fe[cols]], ignore_index=True, sort=False)


def paired_bootstrap_top1(
    df: pd.DataFrame, score_a: str, score_b: str, n_boot: int = 5000, seed: int = 42
) -> dict:
    by = []
    for _rid, g in df.groupby("raceId"):
        wa = g.loc[g[score_a].idxmax(), "winner"] == 1
        wb = g.loc[g[score_b].idxmax(), "winner"] == 1
        by.append((int(wa), int(wb)))
    arr = np.asarray(by)
    delta = (arr[:, 1] - arr[:, 0]).astype(float)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        boots.append(rng.choice(delta, size=len(delta), replace=True).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "delta": float(delta.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_races": len(delta),
    }
