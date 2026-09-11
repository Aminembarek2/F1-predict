"""Race-level evaluation.

Row-level accuracy is meaningless here: every race is one ranking problem with
one winner, so all metrics are computed per race and then averaged over races.

Metrics
-------
``top1``          winner is the highest-scored entry.
``winner_top3``   winner appears in the predicted top three.
``ndcg5``         graded ranking quality at the sharp end.
``spearman``      rank correlation over the whole classified field.
``rank_mae``      mean absolute predicted-vs-actual position error.
``winner_logloss``/``winner_brier``  probabilistic sharpness, using the
                  normalised win probabilities.

Calibration is reported separately because a model can rank well while being
badly over-confident, and an over-confident forecast is worse than useless for
decision making.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import SEED

EPS = 1e-12


def _dcg(relevances: np.ndarray) -> float:
    ranks = np.arange(1, len(relevances) + 1)
    return float(np.sum((2**relevances - 1) / np.log2(ranks + 1)))


def ndcg_at_k(true_positions: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    """NDCG@k with relevance decreasing in true finishing position."""
    n = len(true_positions)
    if n == 0:
        return float("nan")
    # Relevance 5 for the winner down to 0 outside the top 6.
    relevance = np.clip(6 - true_positions, 0, 5).astype(float)
    order = np.argsort(-scores, kind="stable")
    gained = _dcg(relevance[order][:k])
    ideal = _dcg(np.sort(relevance)[::-1][:k])
    return float(gained / ideal) if ideal > EPS else float("nan")


def race_metrics(
    race: pd.DataFrame, score_col: str, prob_col: str | None = None
) -> dict[str, float]:
    """Metrics for a single race.

    ``race`` must hold ``position_order`` (actual classification order),
    ``classified`` and the score column. Higher score = predicted to finish ahead.
    """
    g = race.dropna(subset=[score_col]).copy()
    if g.empty:
        return {}
    # Ties must be broken by something independent of the outcome. Sorting only by
    # score lets the caller's row order decide, and if that order happens to be the
    # finishing order the metric silently reads the answer - a real leak found
    # during development when simulated win probabilities tied at exactly zero.
    tiebreak = g["driver_id"] if "driver_id" in g else g.index.astype(str)
    g = (
        g.assign(_tiebreak=tiebreak.astype(str))
        .sort_values([score_col, "_tiebreak"], ascending=[False, True], kind="mergesort")
        .drop(columns="_tiebreak")
        .reset_index(drop=True)
    )
    g["predicted_rank"] = np.arange(1, len(g) + 1)

    winners = g.loc[(g["position_order"] == 1) & (g["classified"] == 1)]
    if winners.empty:
        return {}
    winner_rank = int(winners["predicted_rank"].iloc[0])

    fin = g.loc[g["classified"] == 1]
    rho = (
        float(spearmanr(fin["predicted_rank"], fin["position_order"]).statistic)
        if len(fin) > 2
        else float("nan")
    )
    mae = float(np.mean(np.abs(g["predicted_rank"] - g["position_order"])))
    nd = ndcg_at_k(g["position_order"].to_numpy(float), g[score_col].to_numpy(float), k=5)

    out = {
        "top1": float(winner_rank == 1),
        "winner_top3": float(winner_rank <= 3),
        "winner_rank": float(winner_rank),
        "spearman": rho,
        "rank_mae": mae,
        "ndcg5": nd,
        "n_entries": float(len(g)),
    }
    if prob_col and prob_col in g:
        raw_p = pd.to_numeric(g[prob_col], errors="coerce")
        if raw_p.isna().all():
            out["winner_logloss"] = float("nan")
            out["winner_brier"] = float("nan")
            return out
        p = raw_p.to_numpy(float)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("probabilities must be finite and in [0, 1]")
        total = p.sum()
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError("win probabilities must sum to one per race")
        is_winner = ((g["position_order"] == 1) & (g["classified"] == 1)).to_numpy()
        out["winner_logloss"] = float(-np.log(max(float(p[is_winner].sum()), EPS)))
        out["winner_brier"] = float(np.mean((p - is_winner.astype(float)) ** 2))
    return out


def evaluate(
    predictions: pd.DataFrame,
    *,
    score_col: str = "score",
    prob_col: str | None = "win_probability",
    race_col: str = "race_id",
) -> pd.DataFrame:
    """Per-race metric table for a set of predictions."""
    rows = []
    for race_id, g in predictions.groupby(race_col):
        m = race_metrics(g, score_col, prob_col)
        if m:
            rows.append({race_col: race_id, **m})
    return pd.DataFrame(rows)


def summarise(per_race: pd.DataFrame, label: str = "model") -> dict[str, float]:
    """Aggregate a per-race metric table into one row."""
    if per_race.empty:
        return {"model": label, "n_races": 0}
    numeric = per_race.select_dtypes("number")
    out: dict[str, float] = {"model": label, "n_races": float(len(per_race))}
    for col in numeric.columns:
        out[col] = float(numeric[col].mean())
    return out


def wilson_interval(successes: int, trials: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (small-sample safe)."""
    if trials <= 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (float(centre - half), float(centre + half))


def paired_bootstrap(
    per_race_a: pd.DataFrame,
    per_race_b: pd.DataFrame,
    metric: str,
    *,
    race_col: str = "race_id",
    n_boot: int = 10_000,
    seed: int = SEED,
    higher_is_better: bool | None = None,
) -> dict[str, float]:
    """Paired bootstrap of ``metric`` differences (B minus A) over shared races.

    Pairing on race removes race difficulty from the comparison, which matters a
    great deal when a season holds only ~20 races.
    """
    merged = (
        per_race_a[[race_col, metric]]
        .merge(
            per_race_b[[race_col, metric]],
            on=race_col,
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if merged.empty:
        return {
            "delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_races": 0,
        }
    delta = (merged[f"{metric}_b"] - merged[f"{metric}_a"]).to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(delta, size=(n_boot, len(delta)), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    if higher_is_better is None:
        higher_is_better = metric not in {"rank_mae", "winner_logloss", "winner_brier", "ece"}
    return {
        "delta": float(np.nanmean(delta)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_better": float((draws > 0).mean() if higher_is_better else (draws < 0).mean()),
        "n_races": int(len(delta)),
    }


def calibration_table(
    predictions: pd.DataFrame,
    *,
    prob_col: str,
    outcome_col: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Reliability table plus expected calibration error for a probability column."""
    df = predictions[[prob_col, outcome_col]].dropna().copy()
    if df.empty:
        return pd.DataFrame()
    df["bin"] = pd.cut(df[prob_col], np.linspace(0, 1, n_bins + 1), include_lowest=True)
    grouped = df.groupby("bin", observed=True).agg(
        n=(outcome_col, "size"),
        predicted=(prob_col, "mean"),
        observed=(outcome_col, "mean"),
    )
    grouped["abs_error"] = (grouped["predicted"] - grouped["observed"]).abs()
    total = grouped["n"].sum()
    grouped["ece_contribution"] = grouped["abs_error"] * grouped["n"] / max(total, 1)
    return grouped.reset_index()


def expected_calibration_error(
    predictions: pd.DataFrame, *, prob_col: str, outcome_col: str, n_bins: int = 10
) -> float:
    """Weighted mean gap between predicted and observed frequency."""
    table = calibration_table(
        predictions, prob_col=prob_col, outcome_col=outcome_col, n_bins=n_bins
    )
    return float(table["ece_contribution"].sum()) if not table.empty else float("nan")


#: Probability columns the forecast publishes, with the outcome each one predicts.
PUBLISHED_PROBABILITIES: tuple[tuple[str, str], ...] = (
    ("win_probability", "winner"),
    ("podium_probability", "podium"),
    ("top10_probability", "top10"),
    ("dnf_probability", "dnf"),
)


def outcome_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """Attach the outcomes implied by the finishing positions.

    Podium and top-10 are always derived here; the winner and retirement columns
    are only filled in when the evaluated frame does not already carry the
    authoritative ones from the results table.
    """
    df = predictions.copy()
    classified = pd.to_numeric(df.get("classified"), errors="coerce").fillna(0).astype(bool)
    position = pd.to_numeric(df.get("position_order"), errors="coerce")
    df["podium"] = (classified & position.le(3)).astype(float)
    df["top10"] = (classified & position.le(10)).astype(float)
    if "winner" not in df:
        df["winner"] = (classified & position.eq(1)).astype(float)
    if "dnf" not in df:
        df["dnf"] = (~classified).astype(float)
    return df


def probability_report(predictions: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame:
    """Calibration of every published probability, not the winner's alone.

    A low pooled winner ECE says nothing about podium, top-10 or retirement
    probabilities, and the uniform forecast can attain an ECE of zero, so each
    outcome is reported separately with its own counts. The interval row reports
    how often the actual finish fell inside the published P10-P90 band; it is a
    pooled descriptive check, not proof of conditional coverage.
    """
    df = outcome_frame(predictions)
    rows = []
    for prob_col, outcome_col in PUBLISHED_PROBABILITIES:
        if prob_col not in df or outcome_col not in df:
            continue
        pair = df[[prob_col, outcome_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if pair.empty:
            continue
        rows.append(
            {
                "outcome": outcome_col,
                "probability_column": prob_col,
                "n": int(len(pair)),
                "predicted_mean": float(pair[prob_col].mean()),
                "observed_rate": float(pair[outcome_col].mean()),
                "ece": expected_calibration_error(
                    pair, prob_col=prob_col, outcome_col=outcome_col, n_bins=n_bins
                ),
                "brier": float(((pair[prob_col] - pair[outcome_col]) ** 2).mean()),
            }
        )
    if {"finish_p10", "finish_p90", "position_order"} <= set(df):
        band = df[["finish_p10", "finish_p90", "position_order"]].apply(
            pd.to_numeric, errors="coerce"
        )
        band = band.dropna()
        if not band.empty:
            inside = band["position_order"].between(band["finish_p10"], band["finish_p90"])
            rows.append(
                {
                    "outcome": "finish_inside_p10_p90",
                    "probability_column": "finish_p10..finish_p90",
                    "n": int(len(band)),
                    "predicted_mean": 0.80,
                    "observed_rate": float(inside.mean()),
                    "ece": float(abs(0.80 - inside.mean())),
                    "brier": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def paired_table(
    per_race: pd.DataFrame,
    *,
    incumbent: str,
    metrics: tuple[str, ...] = ("top1", "ndcg5", "spearman", "rank_mae", "winner_logloss"),
    model_col: str = "model",
) -> pd.DataFrame:
    """Paired bootstrap of every model against one incumbent, race by race."""
    base = per_race.loc[per_race[model_col].eq(incumbent)]
    if base.empty:
        raise ValueError(f"incumbent {incumbent!r} is absent from the metric table")
    rows = []
    for model, group in per_race.groupby(model_col):
        if model == incumbent:
            continue
        for metric in metrics:
            if metric not in group or group[metric].isna().all():
                continue
            rows.append(
                {
                    "model": model,
                    "incumbent": incumbent,
                    "metric": metric,
                    **paired_bootstrap(base, group, metric),
                }
            )
    return pd.DataFrame(rows)
