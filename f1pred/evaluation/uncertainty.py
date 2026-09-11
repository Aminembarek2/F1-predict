"""Dependence sensitivity for paired race comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SEED


def paired_block_bootstrap(
    a: pd.DataFrame,
    b: pd.DataFrame,
    metric: str,
    race_order: pd.DataFrame,
    *,
    block_size: int = 4,
    n_boot: int = 10000,
    seed: int = SEED,
) -> dict:
    """Circular moving blocks within seasons, paired before resampling.

    Preserves short stretches of shared car form; blocks never cross regulation
    seasons. Circular wrapping is a bootstrap convention, not a training split.
    This is a sensitivity analysis, not a cure for four years of development
    data or repeated model selection.
    """
    if block_size < 1 or n_boot < 1:
        raise ValueError("positive block size and bootstrap count required")
    joined = (
        a[["race_id", metric]]
        .merge(b[["race_id", metric]], on="race_id", suffixes=("_a", "_b"), validate="one_to_one")
        .merge(
            race_order[["race_id", "season", "event_index"]],
            on="race_id",
            validate="one_to_one",
            how="left",
        )
    )
    if joined[["season", "event_index"]].isna().any().any():
        raise ValueError("missing race chronology")
    joined = joined.sort_values("event_index").replace([np.inf, -np.inf], np.nan).dropna()
    if joined.empty:
        raise ValueError("no finite paired races")
    rng = np.random.default_rng(seed)
    totals = np.zeros(n_boot)
    for _, group in joined.groupby("season"):
        delta = (group[f"{metric}_b"] - group[f"{metric}_a"]).to_numpy(float)
        n = len(delta)
        width = min(block_size, n)
        starts = rng.integers(0, n, size=(n_boot, int(np.ceil(n / width))))
        indices = ((starts[:, :, None] + np.arange(width)) % n).reshape(n_boot, -1)[:, :n]
        totals += delta[indices].sum(axis=1)
    draws = totals / len(joined)
    lower_is_better = metric in {"winner_logloss", "winner_brier", "rank_mae", "ece"}
    return {
        "delta": float((joined[f"{metric}_b"] - joined[f"{metric}_a"]).mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "p_better": float((draws < 0 if lower_is_better else draws > 0).mean()),
        "n_races": len(joined),
        "block_size": block_size,
    }
