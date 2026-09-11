"""End-to-end prediction pipeline.

Assembles the historical panel, fits the stage-appropriate models for one target
race, and produces a simulated probabilistic forecast.

The pipeline is written so that *every* function takes an explicit
``as_of_index`` derived from the target race. Nothing reads the target race's own
outcome, which is what makes the walk-forward backtest and the live Madrid
forecast run through identical code paths.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DEFAULT_MODEL_CONFIG, STAGE_ORDER, ModelConfig, regime_for
from .features.circuits import (
    build_circuit_profiles,
    circuit_speed_profile,
    measured_overtaking_difficulty,
    nearest_circuits,
    transfer_overtaking_difficulty,
)
from .features.pace import StrengthEstimate, build_observations, entry_strength, fit_strength
from .features.reliability import (
    STREET_CIRCUITS,
    ReliabilityEstimate,
    fit_reliability,
    track_type_multiplier,
)
from .leakage import LeakageError, assert_observed_before, assert_stage_allowed, prediction_cutoff
from .sim.race import SimulationSettings, simulate_race

log = logging.getLogger(__name__)


@dataclass
class Panel:
    """The full historical panel with a global chronological index."""

    results: pd.DataFrame
    qualifying: pd.DataFrame
    schedule: pd.DataFrame
    observations: pd.DataFrame
    event_order: pd.DataFrame
    standings: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Counted on-track passes, when a timing source has supplied them.
    overtakes: pd.DataFrame = field(default_factory=pd.DataFrame)

    def index_of(self, race_id: str) -> float:
        """Chronological index of a race; races not yet run get the next index."""
        match = self.event_order.loc[self.event_order["race_id"] == race_id, "event_index"]
        if len(match):
            return float(match.iloc[0])
        raise ValueError(f"race {race_id!r} is not on the calendar")


def build_panel(
    results: pd.DataFrame,
    qualifying: pd.DataFrame,
    schedule: pd.DataFrame,
    standings: pd.DataFrame | None = None,
    overtakes: pd.DataFrame | None = None,
    race_score: str = "rank",
) -> Panel:
    """Attach a global event index and assemble the observation table."""
    if schedule["race_id"].isna().any() or schedule["race_id"].duplicated().any():
        raise ValueError("calendar race IDs must be present and unique")
    if schedule.duplicated(["season", "round"]).any():
        raise ValueError("calendar season/round pairs must be unique")
    order = (
        schedule[["season", "round", "race_id", "date"]]
        .drop_duplicates()
        .sort_values(["season", "round"])
        .reset_index(drop=True)
    )
    order["event_index"] = np.arange(len(order), dtype=float)

    indexed = []
    for name, frame in (("results", results), ("qualifying", qualifying)):
        if frame.duplicated(["race_id", "driver_id"]).any():
            raise ValueError(f"duplicate race/driver in {name}")
        joined = frame.drop(columns=["event_index"], errors="ignore").merge(
            order[["race_id", "event_index"]], on="race_id", how="left", validate="many_to_one"
        )
        if joined["event_index"].isna().any():
            raise ValueError(f"{name} contains a race absent from the calendar")
        if "circuit_id" not in joined:
            joined = joined.merge(
                schedule[["race_id", "circuit_id"]],
                on="race_id",
                how="left",
                validate="many_to_one",
            )
        event_dates = joined["race_id"].map(schedule.set_index("race_id")["date"])
        joined["historical_available_at"] = pd.to_datetime(event_dates, utc=True) + pd.to_timedelta(
            1, unit="D"
        )
        joined["availability_basis"] = "race_date_plus_one_day"
        if "published_at" in joined:
            # Publication evidence is distinct from download provenance.
            known = joined["published_at"].notna()
            joined.loc[known, "historical_available_at"] = pd.to_datetime(
                joined.loc[known, "published_at"], utc=True, errors="raise"
            )
            joined.loc[known, "availability_basis"] = "published_at"
        indexed.append(joined)
    res, qual = indexed
    obs = build_observations(res, qual, race_score=race_score)
    passes = pd.DataFrame() if overtakes is None else overtakes.copy()
    if not passes.empty:
        # Passes carry the panel's own event index, so a circuit's measured value
        # can be cut off at the target race like every other observation.
        passes["event_index"] = passes["race_id"].map(order.set_index("race_id")["event_index"])
        passes = passes.loc[passes["event_index"].notna()]
    return Panel(
        results=res,
        qualifying=qual,
        schedule=schedule,
        observations=obs,
        event_order=order,
        standings=standings if standings is not None else pd.DataFrame(),
        overtakes=passes,
    )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def baseline_scores(panel: Panel, race_id: str, entries: pd.DataFrame, kind: str) -> pd.Series:
    """Score entries with a simple, transparent baseline.

    Supported baselines
    -------------------
    ``championship``   current drivers' championship order (pre-weekend legal).
    ``last_race``      previous race's classification order (pre-weekend legal).
    ``form5``          negative mean finishing position over the last 5 races,
                       the archived ``PRE_WEEKEND_FORM`` family.
    ``grid``           starting grid of the target race (``POST_QUALI`` only).
    ``qualifying``     qualifying position of the target race (``POST_QUALI`` only).
    """
    idx = panel.index_of(race_id)
    history = panel.results.loc[panel.results["event_index"] < idx]

    if kind == "uniform":
        return pd.Series(0.0, index=entries.index)

    if kind == "championship":
        season = int(entries["season"].iloc[0])
        rnd = int(entries["round"].iloc[0])
        if rnd == 1:
            # At an opener nobody has scored yet. Returning a flat zero makes the
            # baseline degenerate exactly where it is easiest to beat, which
            # flatters anything compared against it, so the previous season's
            # final order is used instead - pre-weekend legal, and the order a
            # person reading a championship table in March would actually have.
            prior = panel.standings.loc[panel.standings["season"].eq(season - 1)]
            if prior.empty:
                return pd.Series(0.0, index=entries.index)
            final = prior.loc[prior["after_round"].eq(prior["after_round"].max())]
            rank = final.set_index("driver_id")["championship_position"]
            return -entries["driver_id"].map(rank).fillna(float(len(final) + 1))
        if panel.standings.empty:
            raise ValueError(
                "official standings missing; ingest standings or use gp_points explicitly"
            )
        prior = panel.standings.loc[
            (panel.standings["season"] == season) & (panel.standings["after_round"] == rnd - 1)
        ]
        if prior.empty:
            raise ValueError(f"official standings missing before {race_id}")
        rank = prior.set_index("driver_id")["championship_position"]
        return -entries["driver_id"].map(rank).fillna(float(len(prior) + 1))

    if kind == "gp_points":
        season = int(entries["season"].iloc[0])
        season_hist = history.loc[history["season"] == season]
        pts = season_hist.groupby("driver_id")["points"].sum()
        return entries["driver_id"].map(pts).fillna(0.0)

    if kind == "last_race":
        if history.empty:
            return pd.Series(0.0, index=entries.index)
        last = history.loc[history["event_index"] == history["event_index"].max()]
        pos = last.set_index("driver_id")["position_order"]
        return -entries["driver_id"].map(pos).fillna(float(len(entries) + 1))

    if kind == "form5":
        recent = history.sort_values("event_index").groupby("driver_id").tail(5)
        mean_pos = recent.groupby("driver_id")["position_order"].mean()
        return -entries["driver_id"].map(mean_pos).fillna(float(len(entries) + 1))

    if kind in {"grid", "qualifying"}:
        col = "grid" if kind == "grid" else "qual_position"
        if col not in entries:
            raise ValueError(f"{kind} baseline requires the corresponding snapshot column")
        pos = pd.to_numeric(entries[col], errors="coerce").replace(0, np.nan)
        return -pos.fillna(float(len(entries) + 1)).rank(method="min")

    raise ValueError(f"unknown baseline {kind!r}")


# ---------------------------------------------------------------------------
# Circuit resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CircuitContext:
    """Everything the simulator needs to know about the venue."""

    circuit_id: str
    overtaking_difficulty: float
    dnf_multiplier: float
    source: str
    analogues: tuple[str, ...] = ()


def resolve_circuit(
    panel: Panel,
    race_id: str,
    circuit_id: str,
    *,
    published_lap_length_m: float | None = None,
    published_pole_time_s: float | None = None,
    is_street: bool | None = None,
    min_history: int = 3,
    measured_overtaking: bool = False,
) -> CircuitContext:
    """Describe the target circuit using only races before it.

    A circuit with at least ``min_history`` previous visits is characterised
    empirically. A brand-new circuit is characterised by transfer: its published
    lap length and projected pole time place it in the derived speed profile
    space, and its nearest same-family neighbours supply the behaviour.
    """
    idx = panel.index_of(race_id)
    history = panel.results.loc[panel.results["event_index"] < idx]
    qual_history = panel.qualifying.loc[panel.qualifying["event_index"] < idx]
    profiles = build_circuit_profiles(history.loc[history["season"] >= 2018])

    street = bool(circuit_id in STREET_CIRCUITS) if is_street is None else bool(is_street)

    counted = pd.DataFrame()
    if measured_overtaking and not panel.overtakes.empty:
        counted = measured_overtaking_difficulty(panel.overtakes, history, as_of_index=idx)
        counted = counted.loc[counted["circuit_id"] == circuit_id]

    own = profiles.loc[profiles["circuit_id"] == circuit_id]
    if len(own) and float(own["n_races"].iloc[0]) >= min_history:
        # A counted pass rate is a direct measurement; the grid-to-finish index is
        # a proxy for the same thing. Prefer the measurement when one exists.
        if len(counted):
            return CircuitContext(
                circuit_id=circuit_id,
                overtaking_difficulty=float(counted["overtaking_difficulty"].iloc[0]),
                dnf_multiplier=float(own["dnf_multiplier"].iloc[0]),
                source="measured_passes",
            )
        return CircuitContext(
            circuit_id=circuit_id,
            overtaking_difficulty=float(own["overtaking_difficulty"].iloc[0]),
            dnf_multiplier=float(own["dnf_multiplier"].iloc[0]),
            source="empirical",
        )

    analogues: tuple[str, ...] = ()
    if published_lap_length_m and published_pole_time_s:
        speed = circuit_speed_profile(
            history.loc[history["season"] >= 2018], qual_history.loc[qual_history["season"] >= 2018]
        )
        target = pd.Series(
            {
                "circuit_id": circuit_id,
                "lap_length_m": float(published_lap_length_m),
                "avg_lap_speed_kmh": 3.6 * published_lap_length_m / published_pole_time_s,
                "field_spread_pct": float(speed["field_spread_pct"].median()),
            }
        )
        family = set(STREET_CIRCUITS) if street else set(speed["circuit_id"]) - set(STREET_CIRCUITS)
        nn = nearest_circuits(
            target, speed, ["avg_lap_speed_kmh", "lap_length_m"], n=4, restrict_to=family
        )
        analogues = tuple(nn["circuit_id"]) if len(nn) else ()

    difficulty, source = transfer_overtaking_difficulty(
        profiles, is_street=street, reference_circuits=list(analogues) or None
    )
    multiplier = track_type_multiplier(history, is_street=street)
    return CircuitContext(
        circuit_id=circuit_id,
        overtaking_difficulty=difficulty,
        dnf_multiplier=multiplier,
        source=source,
        analogues=analogues,
    )


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------


@dataclass
class Forecast:
    """A complete race forecast plus the evidence behind it."""

    race_id: str
    stage: str
    table: pd.DataFrame
    strength: StrengthEstimate
    reliability: ReliabilityEstimate
    circuit: CircuitContext
    diagnostics: dict[str, float]


def forecast_race(
    panel: Panel,
    race_id: str,
    entries: pd.DataFrame,
    *,
    stage: str = "PRE_WEEKEND",
    circuit: CircuitContext | None = None,
    config: ModelConfig = DEFAULT_MODEL_CONFIG,
    settings: SimulationSettings | None = None,
    grid_col: str | None = None,
    cutoff: pd.Timestamp | str | None = None,
    allow_retrospective: bool = False,
) -> Forecast:
    """Produce a probabilistic forecast for one race at one prediction stage.

    Parameters
    ----------
    entries:
        One row per expected starter with ``driver_id``, ``constructor_id``,
        ``season``, ``circuit_id``. At ``POST_QUALI`` it may also carry a grid
        column, named by ``grid_col``.
    """
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown prediction stage {stage!r}")
    if (
        entries.empty
        or entries["driver_id"].isna().any()
        or entries["driver_id"].duplicated().any()
    ):
        raise ValueError("entry list must contain unique, nonmissing drivers")
    scheduled = panel.schedule.loc[panel.schedule["race_id"] == race_id]
    if scheduled.empty:
        raise ValueError(f"{race_id} is not on the calendar")
    for col in ("season", "circuit_id"):
        if (
            entries[col].nunique(dropna=False) != 1
            or entries[col].iloc[0] != scheduled[col].iloc[0]
        ):
            raise ValueError(f"entry {col} does not match the target calendar event")
    if entries["constructor_id"].isna().any():
        raise ValueError("entry constructors must not be missing")
    if cutoff is None:
        if not allow_retrospective:
            raise LeakageError("an explicit prediction cutoff is required")
        start = scheduled.iloc[0].get("race_start_utc", scheduled["date"].iloc[0])
        cutoff = prediction_cutoff(pd.Timestamp(start), stage)
    if STAGE_ORDER[stage] >= STAGE_ORDER["POST_QUALI"]:
        grid_col = grid_col or "grid"
        if grid_col not in entries or entries[grid_col].isna().any():
            raise ValueError("POST_QUALI and RACE_START require a complete grid snapshot")
    if not allow_retrospective:
        for col in ("observed_at", "source"):
            if col not in entries or entries[col].isna().any():
                raise LeakageError(f"entry snapshot requires {col}")
        assert_observed_before(entries, "observed_at", cutoff)
        if grid_col:
            if "grid_observed_at" not in entries:
                raise LeakageError("grid snapshot requires grid_observed_at")
            assert_observed_before(entries, "grid_observed_at", cutoff)
    families = [
        "historical_results",
        "historical_qualifying",
        "team_strength",
        "driver_strength",
        "reliability",
        "track_similarity",
        "calendar",
    ]
    if grid_col is not None:
        families.append("grid")
    assert_stage_allowed(families, stage)

    idx = panel.index_of(race_id)
    reconstructed_history = False
    for history in (panel.results, panel.qualifying):
        prior = history.loc[history["event_index"] < idx]
        # A past result becomes usable when it was published, not when this
        # repository happened to download it: re-fetching a 2019 race today does
        # not turn its outcome into future information. Fetch time is provenance
        # and is recorded separately; publication time is what the gate enforces.
        if "historical_available_at" not in prior:
            raise LeakageError("history lacks publication timestamps; rebuild the panel")
        assert_observed_before(prior, "historical_available_at", cutoff)
        reconstructed_history |= (
            "availability_basis" not in prior
            or not prior["availability_basis"].eq("published_at").all()
        )
    season = int(entries["season"].iloc[0])
    regime = regime_for(season)
    circuit_id = str(entries["circuit_id"].iloc[0])

    strength = fit_strength(
        panel.observations, as_of_index=idx, target_regime=regime, config=config
    )
    reliability = fit_reliability(
        panel.results, as_of_index=idx, target_regime=regime, config=config
    )
    ctx = circuit or resolve_circuit(panel, race_id, circuit_id)

    scored = entry_strength(strength, entries)
    scored["dnf_probability"] = [
        reliability.hazard(d, c, multiplier=ctx.dnf_multiplier)
        for d, c in zip(scored["driver_id"], scored["constructor_id"], strict=True)
    ]
    # The incumbent retains a hand-set observation-count spread. The optional
    # marginal-posterior arm is a recorded flawed experiment: independent draws
    # discard cross-entry covariance. Neither is the state-space candidate.
    obs_weight = scored["driver_obs_weight"].to_numpy(float)
    heuristic = np.where(scored["is_rookie"], 0.45, 0.45 / (1.0 + obs_weight / 3.0))
    if config.strength_spread == "posterior":
        posterior = pd.to_numeric(scored.get("strength_se"), errors="coerce").to_numpy(float)
        scored["strength_sd"] = np.where(np.isfinite(posterior), posterior, heuristic)
    else:
        scored["strength_sd"] = heuristic

    sim = simulate_race(
        scored,
        overtaking_difficulty=ctx.overtaking_difficulty,
        settings=settings or SimulationSettings(n_sims=config.n_simulations, seed=config.seed),
        grid_col=grid_col,
    )
    table = sim.table.copy()
    table["race_id"] = race_id
    table["stage"] = stage
    table["cutoff_utc"] = pd.Timestamp(cutoff).isoformat()
    table["evaluation_scope"] = (
        "retrospective_conditional_field"
        if allow_retrospective
        else "reconstructed_snapshot"
        if reconstructed_history
        else "timestamped_snapshot"
    )
    table["practice_used"] = False
    # Canonical ranking key, selected by paired bootstrap on the 2022-2025 window
    # against the unconditional expected finish, the raw strength and the win
    # probability (see results/scoring_key_selection.csv). Expected position
    # *given the car classifies* wins because it is continuous - so it cannot tie
    # deep in the field - and is not dominated by the retirement tail.
    table["score"] = table["pace_rank_score"]
    return Forecast(
        race_id=race_id,
        stage=stage,
        table=table,
        strength=strength,
        reliability=reliability,
        circuit=ctx,
        diagnostics={**sim.diagnostics, "n_strength_obs": float(strength.n_observations)},
    )
