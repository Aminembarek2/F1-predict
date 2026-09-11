"""Comparison against a bookmaker win market.

A bookmaker board is the strongest publicly available benchmark, but raw quoted
prices are **not** probabilities: the book is deliberately over-round, summing to
more than 1 by the margin the bookmaker keeps. Comparing a model's probabilities
to unadjusted implied prices therefore flatters the model everywhere.

The overround is removed here by proportional normalisation, which is the
standard first-order correction. It slightly over-corrects long shots (the
favourite-longshot bias means the true margin is not uniform across runners), so
the normalised board is treated as a strong reference rather than ground truth.
"""

from __future__ import annotations

import pandas as pd


def american_to_implied(odds: pd.Series) -> pd.Series:
    """Convert American moneyline odds to raw implied probability."""
    o = pd.to_numeric(odds, errors="coerce")
    positive = 100.0 / (o + 100.0)
    negative = -o / (-o + 100.0)
    return positive.where(o > 0, negative)


def remove_overround(implied: pd.Series) -> pd.Series:
    """Normalise a set of implied probabilities to sum to one."""
    total = implied.sum()
    return implied / total if total > 0 else implied


def market_probabilities(
    odds_table: pd.DataFrame, *, odds_col: str = "american_odds"
) -> pd.DataFrame:
    """Return the market board with raw and overround-adjusted probabilities."""
    out = odds_table.copy()
    out["implied_raw"] = american_to_implied(out[odds_col])
    out["overround"] = float(out["implied_raw"].sum())
    out["market_probability"] = remove_overround(out["implied_raw"])
    return out


def compare_to_market(
    forecast: pd.DataFrame,
    market: pd.DataFrame,
    *,
    on: str = "driver_code",
    model_col: str = "win_probability",
) -> pd.DataFrame:
    """Join a model forecast to a normalised market board and report disagreement."""
    board = market_probabilities(market)
    merged = forecast.merge(
        board[[on, "market_probability", "implied_raw", "overround"]], on=on, how="left"
    )
    merged["edge_pp"] = 100.0 * (merged[model_col] - merged["market_probability"])
    merged["model_rank"] = merged[model_col].rank(ascending=False, method="min")
    merged["market_rank"] = merged["market_probability"].rank(ascending=False, method="min")
    merged["rank_disagreement"] = merged["model_rank"] - merged["market_rank"]
    return merged.sort_values(model_col, ascending=False).reset_index(drop=True)


def agreement_metrics(
    compared: pd.DataFrame, *, model_col: str = "win_probability"
) -> dict[str, float]:
    """Summarise how far the model sits from the market."""
    valid = compared.dropna(subset=["market_probability"])
    if valid.empty:
        return {}
    from scipy.stats import spearmanr

    return {
        "spearman_vs_market": float(
            spearmanr(valid[model_col], valid["market_probability"]).statistic
        ),
        "mean_abs_edge_pp": float(valid["edge_pp"].abs().mean()),
        "max_abs_edge_pp": float(valid["edge_pp"].abs().max()),
        "same_favourite": float(
            valid.loc[valid[model_col].idxmax(), "driver_code"]
            == valid.loc[valid["market_probability"].idxmax(), "driver_code"]
        ),
        "overround": float(valid["overround"].iloc[0]),
    }
