"""Experimental long-run practice pace with explicit data-quality controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from ..leakage import assert_observed_before, assert_stage_allowed


def corrected_practice_pace(
    laps: pd.DataFrame,
    *,
    cutoff: str | pd.Timestamp,
    stage: str = "POST_FP2",
    minimum_laps: int = 5,
) -> pd.DataFrame:
    """Estimate relative dry FP2 pace, conditional on uncertain fuel loads.

    Within a session, robust regression adjusts for tyre age, compound and linear
    session evolution. Driver/stint intercepts keep performance differences from
    being attributed entirely to those controls. Fuel is not directly observed;
    uncertainty includes a fixed 0.5 s floor, not an exact fuel correction.
    """
    assert_stage_allowed(["practice_fp2"], stage)
    required = {
        "driver_id",
        "lap_duration",
        "tyre_age",
        "compound",
        "stint_number",
        "observed_at",
        "date_start",
        "is_pit_out_lap",
        "interrupted",
        "rainfall",
    }
    missing = required - set(laps)
    if missing:
        raise ValueError(f"practice inputs missing {sorted(missing)}")
    assert_observed_before(laps, "observed_at", cutoff)
    g = laps.copy()
    for col in ("lap_duration", "tyre_age", "stint_number", "rainfall"):
        g[col] = pd.to_numeric(g[col], errors="coerce")
    g = g.loc[
        g["driver_id"].notna()
        & g["compound"].isin(["SOFT", "MEDIUM", "HARD"])
        & g["lap_duration"].gt(0)
        & g["tyre_age"].ge(2)
        & g["rainfall"].eq(0)
        & g["is_pit_out_lap"].eq(False)
        & g["interrupted"].eq(False)
    ].copy()
    if g.empty:
        return pd.DataFrame(columns=["driver_id", "practice_score", "pace_sd", "n_laps"])
    # Removes slow pit-in/cooldown/traffic laps, using only the current practice
    # session. Exclusion uses a driver/stint median, not the eventual race result.
    key = ["driver_id", "stint_number"]
    median = g.groupby(key)["lap_duration"].transform("median")
    g = g.loc[g["lap_duration"].between(median * 0.95, median * 1.025)].copy()
    g = g.loc[g.groupby(key)["lap_duration"].transform("size").ge(minimum_laps)].copy()
    if g["driver_id"].nunique() < 4:
        return pd.DataFrame(columns=["driver_id", "practice_score", "pace_sd", "n_laps"])
    observed = pd.to_datetime(g["date_start"], utc=True)
    g["session_hours"] = (observed - observed.min()).dt.total_seconds() / 3600
    g["stint_id"] = g["driver_id"].astype(str) + ":" + g["stint_number"].astype(str)
    controls = pd.concat(
        [
            g[["tyre_age", "session_hours"]],
            pd.get_dummies(g["compound"], prefix="compound", drop_first=True, dtype=float),
        ],
        axis=1,
    )
    intercepts = pd.get_dummies(g["stint_id"], prefix="stint", drop_first=True, dtype=float)
    design = pd.concat([controls, intercepts], axis=1)
    y = g["lap_duration"].to_numpy(float)
    model = HuberRegressor(alpha=1.0, max_iter=2000).fit(design, y)
    g["adjusted"] = y - controls.to_numpy(float) @ model.coef_[: len(controls.columns)]
    summary = g.groupby("driver_id").agg(
        pace=("adjusted", "median"), spread=("adjusted", "std"), n_laps=("adjusted", "size")
    )
    scale = max(float(summary["pace"].std()), 0.5)
    summary["practice_score"] = -(summary["pace"] - summary["pace"].median()) / scale
    summary["pace_sd"] = np.sqrt(summary["spread"].fillna(scale) ** 2 / summary["n_laps"] + 0.5**2)
    return summary.reset_index()[["driver_id", "practice_score", "pace_sd", "n_laps"]]
