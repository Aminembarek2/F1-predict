"""Forecasting an upcoming race.

Differs from the backtest in one respect only: the entry list cannot be read from
the result of a race that has not happened. It is taken from the most recent
completed round, which is genuinely pre-weekend information.
"""

from __future__ import annotations

import logging

import pandas as pd

from .config import DEFAULT_MODEL_CONFIG, ROOT, ModelConfig
from .pipeline import CircuitContext, Forecast, Panel, forecast_race, resolve_circuit
from .sim.race import SimulationSettings

log = logging.getLogger(__name__)

SPEC_PATH = ROOT / "data" / "reference" / "circuit_specs.csv"


def expected_entry_list(panel: Panel, season: int, round_no: int) -> pd.DataFrame:
    """Best pre-weekend estimate of who will start: the last completed round's grid.

    Driver changes announced between rounds are not captured automatically; the
    returned frame is the caller's to amend.
    """
    target = f"{season}_{round_no:02d}"
    idx = panel.index_of(target)
    history = panel.results.loc[panel.results["event_index"] < idx]
    if history.empty:
        raise ValueError(f"no completed races before {target}")
    latest = history.loc[history["event_index"] == history["event_index"].max()]
    if int(latest["season"].iloc[0]) != season:
        raise ValueError(
            "season openers require an explicit entry list, including driver/team changes"
        )
    schedule = panel.schedule.loc[panel.schedule["race_id"] == target]
    if schedule.empty:
        raise ValueError(f"{target} is not on the calendar")

    entries = latest[["driver_id", "driver_code", "constructor_id"]].drop_duplicates().copy()
    entries["season"] = season
    entries["round"] = round_no
    entries["race_id"] = target
    entries["circuit_id"] = str(schedule["circuit_id"].iloc[0])
    entries["source"] = "previous_event_projection"
    # The snapshot is only legal from the moment the previous round's results were
    # published, so it is stamped with that time rather than with the time the
    # forecast happens to be produced.
    stamp = "observed_at" if "observed_at" in latest else "historical_available_at"
    if stamp in latest:
        entries["observed_at"] = latest[stamp].max()
    # Neutral ordering: never inherit an ordering derived from an outcome.
    return entries.sort_values("driver_id").reset_index(drop=True)


def published_spec(circuit_id: str) -> dict[str, float | str] | None:
    """Look up a published circuit specification for a venue with no history."""
    if not SPEC_PATH.exists():
        return None
    specs = pd.read_csv(SPEC_PATH)
    row = specs.loc[specs["circuit_id"] == circuit_id]
    return row.iloc[0].to_dict() if len(row) else None


def forecast_upcoming(
    panel: Panel,
    *,
    season: int,
    round_no: int,
    stage: str = "PRE_WEEKEND",
    n_sims: int = 100_000,
    config: ModelConfig = DEFAULT_MODEL_CONFIG,
    entries: pd.DataFrame | None = None,
    projected_pole_s: float | None = None,
    cutoff: pd.Timestamp | str | None = None,
    allow_retrospective: bool = False,
) -> Forecast:
    """Produce a forecast for a future race.

    For a circuit with no history the published specification (lap length, layout
    family) plus a projected pole time places it in the derived circuit-speed
    space, and its nearest same-family neighbours supply the behaviour.
    """
    race_id = f"{season}_{round_no:02d}"
    if entries is None:
        raise ValueError(
            "provide an entry snapshot; previous_event projections must be requested explicitly"
        )
    circuit_id = str(entries["circuit_id"].iloc[0])

    spec = published_spec(circuit_id)
    kwargs: dict[str, object] = {}
    if spec:
        kwargs["published_lap_length_m"] = float(spec["lap_length_m"])
        kwargs["is_street"] = str(spec.get("layout_family", "")).startswith("street")
        pole = projected_pole_s or float(spec.get("projected_pole_s") or 0) or None
        if pole:
            kwargs["published_pole_time_s"] = pole
        log.info("using published spec for %s: %s", circuit_id, spec.get("source"))

    ctx: CircuitContext = resolve_circuit(panel, race_id, circuit_id, **kwargs)  # type: ignore[arg-type]
    log.info(
        "circuit %s: overtaking_difficulty=%.3f dnf_multiplier=%.3f source=%s analogues=%s",
        circuit_id,
        ctx.overtaking_difficulty,
        ctx.dnf_multiplier,
        ctx.source,
        ctx.analogues,
    )
    return forecast_race(
        panel,
        race_id,
        entries,
        stage=stage,
        circuit=ctx,
        config=config,
        settings=SimulationSettings(n_sims=n_sims, seed=config.seed),
        cutoff=cutoff,
        allow_retrospective=allow_retrospective,
    )
