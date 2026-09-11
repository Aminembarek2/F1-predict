"""Circuit characterisation: how much does starting position decide the race?

The single most consequential circuit property for a pre-race forecast is
**overtaking difficulty** - the degree to which the grid, rather than pace,
determines the finishing order. Starting P5 at Monaco is not starting P5 at
Monza, and a simulator that ignores this mis-states every probability.

Two estimators are provided:

``empirical_overtaking_difficulty``
    Measured from history: how strongly grid rank predicts finishing rank among
    classified finishers, plus how little net position change occurs. Requires
    several past visits.

``transfer_overtaking_difficulty``
    For circuits with no history (Madrid 2026), borrows from circuits of the same
    layout family, optionally refined by published geometry.

Availability: ``PRE_WEEKEND`` (history only; no data from the target race).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .reliability import STREET_CIRCUITS

EPS = 1e-9


@dataclass(frozen=True)
class CircuitProfile:
    """Physical and behavioural description of one circuit."""

    circuit_id: str
    overtaking_difficulty: float  # 0 easy .. 1 impossible
    grid_finish_spearman: float
    mean_abs_position_change: float
    dnf_multiplier: float
    is_street: bool
    n_races: int
    source: str  # "empirical" or "transfer:<family>"


def _per_race_overtaking(group: pd.DataFrame) -> tuple[float, float]:
    """Return (grid-finish Spearman, mean |position change|) for one race."""
    g = group.loc[(group["classified"] == 1) & group["grid"].gt(0)]
    if len(g) < 8:
        return (np.nan, np.nan)
    rho = spearmanr(g["grid"], g["position_order"]).statistic
    change = float(np.mean(np.abs(g["grid"] - g["position_order"])))
    return (float(rho), change)


def empirical_overtaking_difficulty(results: pd.DataFrame, *, min_races: int = 3) -> pd.DataFrame:
    """Measure grid-dependence per circuit from completed races.

    A high Spearman correlation between grid and finish plus a low mean absolute
    position change means track position is decisive. The two signals are
    combined into a 0-1 index by rank-normalising each across the calendar,
    which avoids hard-coding an arbitrary scale.

    Only classified finishers are used, because a retirement's finishing slot
    reflects failure rather than overtaking.
    """
    rows: list[dict[str, float | str]] = []
    for (circuit, race_id), g in results.groupby(["circuit_id", "race_id"]):
        rho, change = _per_race_overtaking(g)
        if np.isfinite(rho):
            rows.append({"circuit_id": circuit, "race_id": race_id, "rho": rho, "change": change})
    per_race = pd.DataFrame(rows)
    if per_race.empty:
        return pd.DataFrame(
            columns=[
                "circuit_id",
                "grid_finish_spearman",
                "mean_abs_position_change",
                "n_races",
                "overtaking_difficulty",
            ]
        )

    agg = per_race.groupby("circuit_id").agg(
        grid_finish_spearman=("rho", "median"),
        mean_abs_position_change=("change", "median"),
        n_races=("race_id", "nunique"),
    )
    reliable = agg.loc[agg["n_races"] >= min_races]
    if reliable.empty:
        reliable = agg
    # Rank-normalise both signals over circuits with enough history, then average.
    rho_rank = reliable["grid_finish_spearman"].rank(pct=True)
    change_rank = 1.0 - reliable["mean_abs_position_change"].rank(pct=True)
    difficulty = ((rho_rank + change_rank) / 2.0).clip(0.02, 0.98)
    agg["overtaking_difficulty"] = difficulty
    return agg.reset_index()


def transfer_overtaking_difficulty(
    profiles: pd.DataFrame, *, is_street: bool, reference_circuits: list[str] | None = None
) -> tuple[float, str]:
    """Estimate difficulty for a circuit with no history.

    Parameters
    ----------
    profiles:
        Output of :func:`empirical_overtaking_difficulty`.
    is_street:
        Whether the new venue is a temporary street layout.
    reference_circuits:
        Explicit analogue circuits chosen from published geometry. When given,
        their median difficulty is used and named in the returned source string;
        this is strictly more informative than the layout-family average.

    Returns
    -------
    (difficulty, source)
    """
    have = profiles.loc[profiles["n_races"] >= 3]
    if reference_circuits:
        subset = have.loc[have["circuit_id"].isin(reference_circuits)]
        if len(subset) >= 2:
            return (
                float(subset["overtaking_difficulty"].median()),
                "transfer:analogues=" + ",".join(sorted(subset["circuit_id"])),
            )
    family = have.loc[have["circuit_id"].isin(STREET_CIRCUITS) == is_street]
    if family.empty:
        return (0.5, "transfer:global_default")
    label = "street" if is_street else "permanent"
    return (float(family["overtaking_difficulty"].median()), f"transfer:family={label}")


def build_circuit_profiles(results: pd.DataFrame) -> pd.DataFrame:
    """Join overtaking difficulty and attrition multipliers into one table."""
    from .reliability import circuit_dnf_multipliers

    overtake = empirical_overtaking_difficulty(results)
    attrition = circuit_dnf_multipliers(results)
    out = overtake.merge(
        attrition[["circuit_id", "dnf_multiplier", "circuit_dnf_rate"]],
        on="circuit_id",
        how="outer",
    )
    out["is_street"] = out["circuit_id"].isin(STREET_CIRCUITS)
    return out.sort_values("overtaking_difficulty", ascending=False).reset_index(drop=True)


#: FIA minimum race distance in metres. Monaco is the standing exception, run to
#: 260 km; ignoring this inflates its recovered lap length by ~17%.
RACE_DISTANCE_M = 305_000.0
RACE_DISTANCE_OVERRIDES_M = {"monaco": 260_000.0}


def circuit_speed_profile(
    results: pd.DataFrame, qualifying: pd.DataFrame, *, since_season: int = 2018
) -> pd.DataFrame:
    """Derive lap length and average lap speed per circuit without external specs.

    Lap length is recovered from the regulated race distance and the winner's lap
    count (``length ~ 305 km / laps``); average speed then follows from the pole
    lap time. Both are medians over seasons, so a single wet or shortened race
    cannot distort the profile.

    This keeps circuit characterisation inside the reproducible data pipeline
    instead of depending on hand-copied track specifications.
    """
    res = results.loc[(results["season"] >= since_season) & (results["winner"] == 1)].copy()
    distance = res["circuit_id"].map(RACE_DISTANCE_OVERRIDES_M).fillna(RACE_DISTANCE_M)
    res["lap_length_m"] = distance / res["laps"].clip(lower=1)
    lap_len = res.groupby("circuit_id")["lap_length_m"].median()

    qual = qualifying.loc[qualifying["season"] >= since_season]
    pole = (
        qual.groupby(["circuit_id", "race_id"])["qual_best_s"].min()
        if "circuit_id" in qual
        else None
    )
    if pole is None:
        raise ValueError("qualifying frame must carry circuit_id")
    pole_med = pole.groupby("circuit_id").median()

    profile = pd.concat([lap_len.rename("lap_length_m"), pole_med.rename("pole_time_s")], axis=1)
    profile["avg_lap_speed_kmh"] = 3.6 * profile["lap_length_m"] / profile["pole_time_s"]
    # Spread of the qualifying field: how strongly a circuit separates cars.
    spread = qual.groupby("circuit_id")["qual_gap_pct"].median()
    profile["field_spread_pct"] = spread
    return profile.reset_index()


def nearest_circuits(
    target: pd.Series,
    profiles: pd.DataFrame,
    feature_cols: list[str],
    *,
    n: int = 4,
    restrict_to: set[str] | None = None,
) -> pd.DataFrame:
    """Find the circuits most similar to ``target`` in standardised feature space.

    Standardisation is essential: raw cosine similarity between all-positive
    physical quantities is close to 1 for every pair and carries almost no
    information. Distances here are computed on z-scores, so "similar" means
    "similar relative to how much circuits actually vary".
    """
    pool = profiles.dropna(subset=feature_cols).copy()
    if restrict_to is not None:
        pool = pool.loc[pool["circuit_id"].isin(restrict_to)]
    if pool.empty:
        return pool
    mu = pool[feature_cols].mean()
    sd = pool[feature_cols].std(ddof=0).replace(0, 1.0)
    z_pool = (pool[feature_cols] - mu) / sd
    z_target = (target[feature_cols].astype(float) - mu) / sd
    pool["distance"] = np.sqrt(((z_pool - z_target) ** 2).sum(axis=1))
    return pool.sort_values("distance").head(n).reset_index(drop=True)


def measured_overtaking_rate(
    overtakes: pd.DataFrame, results: pd.DataFrame, *, min_races: int = 2
) -> pd.DataFrame:
    """Count the passes that actually happened, per circuit, per racing lap.

    :func:`empirical_overtaking_difficulty` infers overtaking from how much the
    finishing order resembles the grid. That is a proxy, and it confounds three
    different things: a circuit where passing is hard, a race where nobody tried,
    and a race where the fast cars started at the front anyway.

    A timing feed reports each pass as an event, so the quantity can simply be
    counted. Normalising by racing laps - the winner's lap count times the
    classified field - makes a 78-lap Monaco comparable with a 44-lap Spa.
    """
    required = {"race_id", "circuit_id"}
    if overtakes.empty or not required <= set(overtakes):
        return pd.DataFrame(columns=["circuit_id", "overtakes_per_100_laps", "n_races"])
    per_race = overtakes.groupby(["circuit_id", "race_id"]).size().rename("passes").reset_index()

    scale = (
        results.loc[results["classified"].eq(1)]
        .groupby("race_id")
        .agg(laps=("laps", "max"), field=("driver_id", "size"))
    )
    per_race = per_race.merge(scale, left_on="race_id", right_index=True, how="inner")
    racing_laps = (per_race["laps"] * per_race["field"]).clip(lower=1)
    per_race["overtakes_per_100_laps"] = 100.0 * per_race["passes"] / racing_laps

    agg = per_race.groupby("circuit_id").agg(
        overtakes_per_100_laps=("overtakes_per_100_laps", "median"),
        total_passes=("passes", "sum"),
        n_races=("race_id", "nunique"),
    )
    return agg.loc[agg["n_races"] >= min_races].sort_values("overtakes_per_100_laps").reset_index()


def pit_lane_loss(pit: pd.DataFrame, *, min_stops: int = 10) -> pd.DataFrame:
    """Median pit-lane time loss per circuit, in seconds.

    The simulator prices strategy luck at a flat 2.0 grid slots everywhere. The
    real cost of a stop varies by more than ten seconds between circuits, and it
    is measured on every stop that has ever been made.
    """
    if pit.empty or "pit_duration" not in pit:
        return pd.DataFrame(columns=["circuit_id", "pit_lane_seconds", "n_stops"])
    clean = pit.copy()
    clean["pit_duration"] = pd.to_numeric(clean["pit_duration"], errors="coerce")
    # A stop under 12 s is a timing artefact and one over 60 s is a repair, not
    # a strategy call; neither describes the routine cost of stopping here.
    clean = clean.loc[clean["pit_duration"].between(12, 60)]
    agg = clean.groupby("circuit_id").agg(
        pit_lane_seconds=("pit_duration", "median"), n_stops=("pit_duration", "size")
    )
    return agg.loc[agg["n_stops"] >= min_stops].sort_values("pit_lane_seconds").reset_index()


def measured_overtaking_difficulty(
    overtakes: pd.DataFrame,
    results: pd.DataFrame,
    *,
    as_of_index: float,
    min_races: int = 2,
) -> pd.DataFrame:
    """Grid-dependence per circuit from counted passes, using only earlier races.

    The rate is rank-normalised across the circuits visible at the cutoff and
    inverted, so the output shares a scale and a direction with
    :func:`empirical_overtaking_difficulty`: 0 means pace decides, 1 means track
    position does. Ranking rather than rescaling keeps one freak race from
    stretching the whole calendar.

    Only passes from races before ``as_of_index`` count, so a circuit's value at
    race *R* can never have been informed by *R* itself.
    """
    empty = pd.DataFrame(columns=["circuit_id", "overtaking_difficulty", "n_races"])
    if overtakes.empty or "event_index" not in overtakes:
        return empty
    prior = overtakes.loc[overtakes["event_index"] < as_of_index]
    rate = measured_overtaking_rate(prior, results, min_races=min_races)
    if len(rate) < 2:
        return empty
    # Ascending rank of the pass rate: fewest passes -> highest difficulty.
    difficulty = 1.0 - rate["overtakes_per_100_laps"].rank(pct=True)
    out = rate[["circuit_id", "n_races"]].copy()
    # rank(pct=True) puts the easiest circuit at exactly 0; shift off the
    # boundary so no venue is modelled as pure pace with zero track position.
    out["overtaking_difficulty"] = difficulty.clip(lower=0.02, upper=0.98)
    return out.reset_index(drop=True)
