"""Leakage control: what each feature may see, and when.

Two independent guards are provided because they catch different mistakes:

``assert_stage_allowed``
    Catches *design* leakage - using a feature family that does not exist yet at
    the prediction snapshot (e.g. grid position at ``PRE_WEEKEND``).

``assert_observed_before``
    Catches *plumbing* leakage - a row whose observation timestamp is at or after
    the prediction cut-off, regardless of which family it belongs to.

A third helper, :func:`shifted_expanding`, is the only sanctioned way to build a
rolling historical statistic, because it shifts before aggregating.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import POST_RACE, STAGE_ORDER


@dataclass(frozen=True)
class FeatureSpec:
    """Catalog entry for one feature family."""

    family: str
    earliest_stage: str
    source: str
    formula: str
    units: str
    missing_policy: str


#: The authoritative availability catalog. Anything absent is treated as POST_RACE
#: (i.e. illegal) so that forgetting to register a feature fails closed.
FEATURE_CATALOG: dict[str, FeatureSpec] = {
    f.family: f
    for f in [
        FeatureSpec(
            "calendar",
            "PRE_WEEKEND",
            "Jolpica schedule",
            "static calendar fields",
            "-",
            "none expected",
        ),
        FeatureSpec(
            "track_geometry",
            "PRE_WEEKEND",
            "MultiViewer / circuit spec",
            "curvature + corner classification from centreline",
            "1/m",
            "fallback to track-type prior",
        ),
        FeatureSpec(
            "track_similarity",
            "PRE_WEEKEND",
            "derived",
            "correlation distance in standardised track space",
            "-",
            "uniform weights",
        ),
        FeatureSpec(
            "historical_results",
            "PRE_WEEKEND",
            "Jolpica results (races < R)",
            "shifted rolling aggregates",
            "various",
            "shrink to field mean",
        ),
        FeatureSpec(
            "historical_qualifying",
            "PRE_WEEKEND",
            "Jolpica qualifying (races < R)",
            "shifted gap-to-pole aggregates",
            "%",
            "shrink to team mean",
        ),
        FeatureSpec(
            "team_strength",
            "PRE_WEEKEND",
            "derived",
            "recency-weighted ridge team effect",
            "s/lap-equivalent",
            "regime prior",
        ),
        FeatureSpec(
            "driver_strength",
            "PRE_WEEKEND",
            "derived",
            "teammate-adjusted driver effect",
            "s/lap-equivalent",
            "rookie prior",
        ),
        FeatureSpec(
            "reliability",
            "PRE_WEEKEND",
            "Jolpica status (races < R)",
            "Beta-smoothed shifted DNF rate",
            "probability",
            "Beta prior",
        ),
        FeatureSpec(
            "weather_forecast",
            "PRE_WEEKEND",
            "Open-Meteo forecast issued before cut-off",
            "forecast hourly fields at race hour",
            "degC / mm / m/s",
            "climatology",
        ),
        FeatureSpec(
            "championship_state",
            "PRE_WEEKEND",
            "Jolpica standings after round R-1",
            "points and rank",
            "points",
            "zero at season start",
        ),
        FeatureSpec(
            "practice_fp1",
            "POST_FP1",
            "OpenF1 / FastF1 laps",
            "track-evolution + fuel corrected pace",
            "s",
            "drop",
        ),
        FeatureSpec(
            "practice_fp2",
            "POST_FP2",
            "OpenF1 / FastF1 laps",
            "single-lap + long-run corrected pace",
            "s",
            "drop",
        ),
        FeatureSpec(
            "practice_fp3",
            "POST_FP3",
            "OpenF1 / FastF1 laps",
            "single-lap corrected pace",
            "s",
            "drop",
        ),
        FeatureSpec(
            "telemetry_practice",
            "POST_FP1",
            "OpenF1 car_data / location",
            "corner braking/traction/aero indices",
            "m/s^2, g",
            "drop corner",
        ),
        FeatureSpec(
            "qualifying",
            "POST_QUALI",
            "Jolpica / OpenF1 qualifying",
            "gap to pole, session position",
            "%",
            "drop",
        ),
        FeatureSpec(
            "grid",
            "POST_QUALI",
            "OpenF1 starting grid + penalties",
            "post-penalty grid slot",
            "position",
            "use qualifying position",
        ),
        FeatureSpec(
            "race_laps", POST_RACE, "OpenF1 laps of race R", "-", "-", "never legal pre-race"
        ),
        FeatureSpec(
            "race_weather_observed",
            POST_RACE,
            "OpenF1 weather of race R",
            "-",
            "-",
            "never legal pre-race",
        ),
        FeatureSpec(
            "race_pitstops", POST_RACE, "OpenF1 pit of race R", "-", "-", "never legal pre-race"
        ),
        FeatureSpec(
            "race_result", POST_RACE, "Jolpica results of race R", "-", "-", "never legal pre-race"
        ),
    ]
}


class LeakageError(ValueError):
    """Raised when a feature would use information from the future."""


def earliest_stage(family: str) -> str:
    """Stage at which ``family`` first becomes legal; unknown families fail closed."""
    spec = FEATURE_CATALOG.get(family)
    return spec.earliest_stage if spec else POST_RACE


def is_allowed(family: str, stage: str) -> bool:
    """True when ``family`` may be used at prediction snapshot ``stage``."""
    need = earliest_stage(family)
    if need == POST_RACE:
        return False
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown prediction stage {stage!r}")
    return STAGE_ORDER[need] <= STAGE_ORDER[stage]


def assert_stage_allowed(families: list[str] | tuple[str, ...], stage: str) -> None:
    """Raise :class:`LeakageError` if any family is unavailable at ``stage``."""
    illegal = [f for f in families if not is_allowed(f, stage)]
    if illegal:
        raise LeakageError(
            f"stage {stage}: {illegal} require {[earliest_stage(f) for f in illegal]}"
        )


def assert_observed_before(df: pd.DataFrame, observed_col: str, cutoff: pd.Timestamp | str) -> None:
    """Raise if any row was observed at or after the prediction cut-off."""
    observed = pd.to_datetime(df[observed_col], utc=True, errors="coerce")
    cut = pd.Timestamp(cutoff)
    if cut.tzinfo is None:
        cut = cut.tz_localize("UTC")
    if pd.isna(cut) or observed.isna().any():
        raise LeakageError(f"invalid or missing timestamp in {observed_col} or cutoff")
    bad = int((observed >= cut).sum())
    if bad:
        raise LeakageError(f"{bad} rows in {observed_col} are at/after cutoff {cut.isoformat()}")


def shifted_expanding(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    order_col: str,
    *,
    window: int | None = None,
    min_periods: int = 1,
    stat: str = "mean",
) -> pd.Series:
    """Rolling historical statistic that always excludes the current row.

    The frame is sorted by ``order_col`` inside each group, shifted by one, and
    only then aggregated - so a value for race R never contains race R itself.

    Parameters
    ----------
    window:
        Number of prior observations to use; ``None`` means expanding.
    stat:
        Any name accepted by pandas rolling/expanding (``mean``, ``sum``, ``std``...).
    """
    if df[group_cols + [order_col]].isna().any().any():
        raise ValueError("aggregation keys must not be missing")
    if df.duplicated(group_cols + [order_col]).any():
        raise LeakageError("aggregate to one row per group/event before shifting")
    ordered = df.sort_values(group_cols + [order_col])
    grouped = ordered.groupby(group_cols, sort=False)[value_col]
    shifted = grouped.shift(1)
    roller = shifted.groupby([ordered[c] for c in group_cols], sort=False)
    if window is None:
        agg = roller.expanding(min_periods=min_periods).agg(stat)
    else:
        agg = roller.rolling(window, min_periods=min_periods).agg(stat)
    agg = agg.reset_index(level=list(range(len(group_cols))), drop=True)
    return agg.reindex(df.index).astype(float)


def prediction_cutoff(race_start_utc: pd.Timestamp, stage: str) -> pd.Timestamp:
    """Latest timestamp whose data may be used, given the snapshot.

    Session offsets are conservative: they assume the standard Friday-to-Sunday
    European format, so the cut-off is never later than the true session end.
    """
    start = pd.Timestamp(race_start_utc)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    offsets_hours = {
        "PRE_WEEKEND": -60.0,  # Friday morning, before FP1
        "POST_FP1": -50.0,
        "POST_FP2": -46.0,
        "POST_FP3": -27.0,
        "POST_QUALI": -23.0,
        "RACE_START": 0.0,
    }
    if stage not in offsets_hours:
        raise ValueError(f"unknown prediction stage {stage!r}")
    return start + pd.to_timedelta(offsets_hours[stage], unit="h")
