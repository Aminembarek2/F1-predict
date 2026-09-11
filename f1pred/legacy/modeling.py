from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def softmax_by_race(scores, race_ids, temperature=1.0):
    s = np.asarray(scores, float)
    r = np.asarray(race_ids)
    out = np.zeros(len(s), float)
    t = max(float(temperature), 1e-6)
    for rid in pd.unique(r):
        idx = np.where(r == rid)[0]
        z = s[idx] / t
        z = z - np.nanmax(z)
        e = np.exp(z)
        out[idx] = e / e.sum()
    return out


def monte_carlo_race(
    drivers: pd.DataFrame,
    n_sims: int = 10000,
    seed: int = 42,
    pace_col="pace_score",
    dnf_prob_col="dnf_prob",
    strategy_sd_col="strategy_sd",
) -> pd.DataFrame:
    """Simple transparent simulator: latent pace + strategy noise, then independent DNF event."""
    rng = np.random.default_rng(seed)
    d = drivers.reset_index(drop=True).copy()
    n = len(d)
    pace = pd.to_numeric(d[pace_col], errors="coerce").fillna(0).to_numpy(float)
    dnf = (
        pd.to_numeric(d.get(dnf_prob_col, 0.05), errors="coerce")
        .fillna(0.05)
        .clip(0, 0.95)
        .to_numpy(float)
    )
    strat = (
        pd.to_numeric(d.get(strategy_sd_col, 0.15), errors="coerce")
        .fillna(0.15)
        .clip(lower=0)
        .to_numpy(float)
    )
    wins = np.zeros(n)
    pods = np.zeros(n)
    top10 = np.zeros(n)
    dnfs = np.zeros(n)
    for _ in range(int(n_sims)):
        perf = pace + rng.normal(0, strat, n)
        retired = rng.random(n) < dnf
        dnfs += retired
        # Retirements ranked behind finishers, with their latent performance only tie-breaking.
        score = perf - retired.astype(float) * 1e6
        order = np.argsort(-score)
        wins[order[0]] += 1
        pods[order[: min(3, n)]] += 1
        top10[order[: min(10, n)]] += 1
    out = d.copy()
    out["win_probability"] = wins / n_sims
    out["podium_probability"] = pods / n_sims
    out["top10_probability"] = top10 / n_sims
    out["dnf_probability_sim"] = dnfs / n_sims
    return out


def fit_dnf_classifier(df: pd.DataFrame, features: list[str], target="dnf"):
    d = df[features + [target]].apply(pd.to_numeric, errors="coerce").dropna()
    if d[target].nunique() < 2:
        raise ValueError("DNF target requires both classes")
    m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
        d[features], d[target].astype(int)
    )
    return m


def race_ranking_metrics(df: pd.DataFrame, score_col="score"):
    from scipy.stats import spearmanr
    from sklearn.metrics import ndcg_score

    rows = []
    for _, g in df.groupby("raceId"):
        g = g.sort_values(score_col, ascending=False).copy()
        g["pred_rank"] = np.arange(1, len(g) + 1)
        if "winner" in g and g.winner.sum() == 1:
            wr = int(g.loc[g.winner == 1, "pred_rank"].iloc[0])
        else:
            wr = np.nan
        rho = (
            spearmanr(g.pred_rank, g.finish_position, nan_policy="omit").statistic
            if "finish_position" in g
            else np.nan
        )
        mae = np.mean(np.abs(g.pred_rank - g.finish_position)) if "finish_position" in g else np.nan
        if {"relevance", score_col}.issubset(g.columns):
            nd = ndcg_score([g.relevance.to_numpy()], [g[score_col].to_numpy()], k=min(5, len(g)))
        else:
            nd = np.nan
        rows.append(
            [
                wr == 1 if np.isfinite(wr) else np.nan,
                wr <= 3 if np.isfinite(wr) else np.nan,
                rho,
                mae,
                nd,
            ]
        )
    a = np.asarray(rows, float)
    return {
        "n_races": len(a),
        "top1": np.nanmean(a[:, 0]),
        "winner_top3": np.nanmean(a[:, 1]),
        "spearman": np.nanmean(a[:, 2]),
        "rank_mae": np.nanmean(a[:, 3]),
        "ndcg5": np.nanmean(a[:, 4]),
    }


def fit_xgb_ranker(train: pd.DataFrame, test: pd.DataFrame, features: list[str], label="relevance"):
    import xgboost as xgb

    tr = train.sort_values(["raceId", "driverId"]).copy()
    te = test.sort_values(["raceId", "driverId"]).copy()
    Xtr = tr[features].apply(pd.to_numeric, errors="coerce")
    Xte = te[features].apply(pd.to_numeric, errors="coerce")
    model = xgb.XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@5",
        tree_method="hist",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2,
        min_child_weight=3,
        lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(Xtr, tr[label].astype(float), qid=tr.raceId.to_numpy(), verbose=False)
    te["score"] = model.predict(Xte)
    return model, te


def temperature_calibrate(scores, race_ids, winners, grid=None):
    grid = np.geomspace(0.1, 6, 60) if grid is None else np.asarray(grid, float)
    best = (1.0, np.inf)
    for t in grid:
        p = softmax_by_race(scores, race_ids, t)
        losses = []
        for rid in pd.unique(race_ids):
            idx = np.where(np.asarray(race_ids) == rid)[0]
            y = np.asarray(winners)[idx]
            pp = p[idx]
            if y.sum() == 1:
                losses.append(-math.log(max(pp[y.astype(bool)][0], 1e-12)))
        loss = np.mean(losses) if losses else np.inf
        if loss < best[1]:
            best = (float(t), float(loss))
    return {"temperature": best[0], "log_loss": best[1]}
