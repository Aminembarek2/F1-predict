"""Race pace measured in time, not in finishing order.

The incumbent scores a race result by rank-transforming the finishing position,
so the model learns that a car finished first and never that it won by twenty
seconds. In a sport where a dominant car wins by half a minute and a close one by
four tenths, that discards the main signal about *how much* quicker a car is.

Every input needed to fix it is already ingested:

``total_ms``
    Total race time, published for every car still on the lead lap. The per-lap
    gap to the winner is a direct measurement of race pace over a full stint
    profile, and it is the cleanest pace signal the sport produces.
``fastest_lap_s``
    One lap, available for every classified finisher including lapped cars. It is
    noisier - fuel load, tyre age and the moment it was set all move it - so it is
    the fallback rather than the primary.

Both are converted to a percentage gap and divided by the same scale
:func:`f1pred.features.pace.qualifying_pace_score` uses, so a qualifying
observation and a race observation land on one latent scale that means something
physical: roughly one unit per 0.75% of lap time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Percentage-gap units per point of latent strength, shared with qualifying.
PACE_SCALE_PCT = 0.75

#: A per-lap gap beyond this is a car with a problem, not a slow car. Kept wide
#: enough to include genuinely uncompetitive machinery in a bad year.
MAX_GAP_PCT = 8.0


def race_pace_gap_pct(results: pd.DataFrame) -> pd.DataFrame:
    """Per-race percentage pace deficit for every classified finisher.

    Returns the input rows with ``pace_gap_pct`` and ``pace_source``. Cars on the
    lead lap are measured on total race time; the rest fall back to their fastest
    lap. A row with neither measurement gets ``NaN`` and is dropped downstream
    rather than imputed.
    """
    df = results.loc[results["classified"].eq(1)].copy()
    if df.empty:
        return df.assign(pace_gap_pct=np.nan, pace_source="none")

    laps = pd.to_numeric(df["laps"], errors="coerce")
    total = pd.to_numeric(df["total_ms"], errors="coerce")
    fastest = pd.to_numeric(df["fastest_lap_s"], errors="coerce")

    # A classified car is on the lead lap if it completed the winner's distance.
    lead_lap_distance = laps.groupby(df["race_id"]).transform("max")
    on_lead_lap = laps.eq(lead_lap_distance) & total.notna()

    per_lap = (total / laps.clip(lower=1)).where(on_lead_lap)
    best_per_lap = per_lap.groupby(df["race_id"]).transform("min")
    total_gap = 100.0 * (per_lap - best_per_lap) / best_per_lap

    best_fastest = fastest.groupby(df["race_id"]).transform("min")
    fastest_gap = 100.0 * (fastest - best_fastest) / best_fastest

    gap = total_gap.where(total_gap.notna(), fastest_gap)
    df["pace_source"] = np.where(
        total_gap.notna(), "race_time", np.where(fastest_gap.notna(), "fastest_lap", "none")
    )
    # A negative gap is arithmetically impossible and signals a corrupt row.
    df["pace_gap_pct"] = gap.where(gap.between(0.0, MAX_GAP_PCT))
    return df


def race_pace_score(results: pd.DataFrame) -> pd.DataFrame:
    """Higher-is-better race pace on the shared latent scale.

    Centred within each race, because the scale is a *relative* pace measurement:
    a slow race and a fast race at the same circuit must not shift every driver.
    """
    df = race_pace_gap_pct(results)
    if df.empty:
        return df.assign(score=np.nan)
    score = -df["pace_gap_pct"] / PACE_SCALE_PCT
    df["score"] = score - score.groupby(df["race_id"]).transform("mean")
    return df


def coverage(results: pd.DataFrame) -> pd.DataFrame:
    """How much of the field each pace source actually covers, by season.

    Reported rather than assumed: if the lead-lap measurement is thin in a
    season, the fallback is carrying that season and the scale is noisier there.
    """
    df = race_pace_gap_pct(results)
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby("season")
    return pd.DataFrame(
        {
            "classified": grouped.size(),
            "race_time": grouped["pace_source"].apply(lambda s: (s == "race_time").sum()),
            "fastest_lap": grouped["pace_source"].apply(lambda s: (s == "fastest_lap").sum()),
            "unusable": grouped["pace_gap_pct"].apply(lambda s: s.isna().sum()),
            "median_gap_pct": grouped["pace_gap_pct"].median(),
        }
    ).reset_index()
