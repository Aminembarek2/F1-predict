"""Walk-forward backtesting.

Protocol
--------
Races are predicted strictly in chronological order. For race *R* the model may
use every race before *R* and nothing else; there is no random splitting and no
re-fitting on future data. This is the only validation design that reflects how
the system would actually have been used.

The same loop evaluates the strength model and every baseline, so comparisons
are paired race-by-race and can be bootstrapped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import DEFAULT_MODEL_CONFIG, STAGE_ORDER, ModelConfig
from ..pipeline import Panel, baseline_scores, forecast_race, resolve_circuit
from ..sim.race import SimulationSettings
from .metrics import evaluate, summarise, wilson_interval

log = logging.getLogger(__name__)


def race_entries(panel: Panel, race_id: str) -> pd.DataFrame:
    """Retrospective evaluation rows, never a verified pre-weekend snapshot.

    The outcome columns are used **only** by the evaluation layer; the model
    never sees them. Keeping them on the same frame avoids a fragile re-join.
    """
    df = panel.results.loc[
        (panel.results["race_id"] == race_id) & (panel.results["started"] == 1)
    ].copy()
    # Sort by driver_id, never by outcome: the raw results frame arrives in
    # finishing order, and any downstream tie would otherwise inherit the answer.
    return (
        df[
            [
                "race_id",
                "season",
                "round",
                "circuit_id",
                "driver_id",
                "driver_code",
                "constructor_id",
                "grid",
                "position_order",
                "classified",
                "dnf",
                "winner",
            ]
        ]
        .sort_values("driver_id", kind="mergesort")
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class FieldAlignment:
    """One race's outcome, restated on the field the forecast actually named."""

    truth: pd.DataFrame
    named: int
    started: int
    did_not_start: int
    unforeseen: int
    scoreable: bool


def align_snapshot_to_result(entries: pd.DataFrame, truth: pd.DataFrame) -> FieldAlignment:
    """Score a forecast against the race on the field it named in advance.

    A pre-weekend snapshot is a forecast of who will start, and it is sometimes
    wrong. The declared policy, applied identically to every model:

    * a named driver who started is scored on the result;
    * a named driver who did not start is scored behind every driver who did,
      because from the forecast's point of view that entry did not complete the
      race - the error belongs to the forecast, not to the evaluation;
    * a driver who started without being named is excluded and counted, since no
      model was asked to rank an entry nobody knew about;
    * a race whose winner was never named is **unscoreable** and is dropped, so a
      missed winner can never be recorded as a correct one.

    Positions are re-ranked densely within the evaluated field. Ties among
    non-starters break on ``driver_id``, never on anything derived from the
    result.
    """
    named = list(dict.fromkeys(entries["driver_id"]))
    actual = truth.set_index("driver_id")
    started = [d for d in named if d in actual.index]
    absent = sorted(d for d in named if d not in actual.index)
    unforeseen = [d for d in actual.index if d not in set(named)]

    winners = truth.loc[truth["winner"].eq(1), "driver_id"]
    scoreable = len(winners) == 1 and winners.iloc[0] in set(started)

    ranked = actual.loc[started].sort_values("position_order", kind="mergesort")
    rows = [
        {
            "driver_id": driver,
            "position_order": rank,
            "classified": int(ranked.loc[driver, "classified"]),
            "dnf": int(ranked.loc[driver, "dnf"]),
            "winner": int(ranked.loc[driver, "winner"]),
        }
        for rank, driver in enumerate(ranked.index, start=1)
    ]
    rows += [
        {
            "driver_id": driver,
            "position_order": len(started) + offset,
            "classified": 0,
            "dnf": 1,
            "winner": 0,
        }
        for offset, driver in enumerate(absent, start=1)
    ]
    return FieldAlignment(
        truth=pd.DataFrame(rows),
        named=len(named),
        started=len(started),
        did_not_start=len(absent),
        unforeseen=len(unforeseen),
        scoreable=scoreable,
    )


def walk_forward(
    panel: Panel,
    race_ids: list[str],
    *,
    stage: str = "PRE_WEEKEND",
    config: ModelConfig = DEFAULT_MODEL_CONFIG,
    n_sims: int = 20_000,
    baselines: tuple[str, ...] = ("gp_points", "last_race", "form5", "uniform"),
    include_grid_reference: bool = False,
    allow_retrospective: bool = False,
    snapshots: dict[str, pd.DataFrame] | None = None,
    cutoffs: dict[str, str] | None = None,
    measured_overtaking: bool = False,
    settings: SimulationSettings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest the strength model and baselines over ``race_ids``.

    Returns
    -------
    (predictions, per_race_metrics)
        ``predictions`` is long-form: one row per race, entry and model.
    """
    if not allow_retrospective and (snapshots is None or cutoffs is None):
        raise ValueError(
            "strict backtests require entry snapshots and cutoffs; use allow_retrospective explicitly for conditional diagnostics"
        )
    settings = settings or SimulationSettings(n_sims=n_sims, seed=config.seed)
    frames: list[pd.DataFrame] = []

    skipped: list[dict] = []
    for race_id in race_ids:
        truth = race_entries(panel, race_id)
        if not allow_retrospective and race_id not in snapshots:
            skipped.append({"race_id": race_id, "reason": "no_snapshot"})
            continue
        entries = truth.copy() if allow_retrospective else snapshots[race_id].copy()
        if entries.empty or truth["winner"].sum() != 1:
            log.debug("skipping %s (no unique winner among starters)", race_id)
            continue
        alignment = None
        if not allow_retrospective:
            alignment = align_snapshot_to_result(entries, truth)
            if not alignment.scoreable:
                # The winner was never on the entry list, so no ranking of that
                # list can be right about it. Dropping the race is the only
                # alternative to scoring a miss as a hit.
                skipped.append({"race_id": race_id, "reason": "winner_not_named"})
                continue
            truth = truth[["driver_id"]].merge(alignment.truth, on="driver_id", how="right")
        ctx = resolve_circuit(
            panel,
            race_id,
            str(entries["circuit_id"].iloc[0]),
            measured_overtaking=measured_overtaking,
        )
        # From POST_QUALI the grid is known, so it is supplied rather than simulated.
        grid_col = "grid" if STAGE_ORDER[stage] >= STAGE_ORDER["POST_QUALI"] else None
        fc = forecast_race(
            panel,
            race_id,
            entries,
            stage=stage,
            circuit=ctx,
            config=config,
            settings=settings,
            grid_col=grid_col,
            cutoff=None if cutoffs is None else cutoffs[race_id],
            allow_retrospective=allow_retrospective,
        )
        model_rows = fc.table.copy()
        if not allow_retrospective and alignment is not None:
            outcomes = ["position_order", "classified", "dnf", "winner"]
            model_rows = model_rows.drop(columns=outcomes, errors="ignore").merge(
                truth[["driver_id"] + outcomes], on="driver_id", validate="one_to_one"
            )
            entries = entries.drop(columns=outcomes, errors="ignore").merge(
                truth[["driver_id"] + outcomes], on="driver_id", validate="one_to_one"
            )
            # How wrong the entry forecast itself was, carried on every row so no
            # reader has to take the field construction on trust.
            for frame in (model_rows, entries):
                frame["field_named"] = alignment.named
                frame["field_did_not_start"] = alignment.did_not_start
                frame["field_unforeseen"] = alignment.unforeseen
        model_rows["model"] = "strength_mc"
        frames.append(model_rows)

        for kind in baselines:
            b = entries.copy()
            b["score"] = baseline_scores(panel, race_id, b, kind)
            b["win_probability"] = 1 / len(b) if kind == "uniform" else np.nan
            b["model"] = kind
            b["race_id"] = race_id
            b["stage"] = stage
            b["evaluation_scope"] = model_rows["evaluation_scope"].iloc[0]
            frames.append(b)

        if include_grid_reference:
            if not allow_retrospective and stage not in {"POST_QUALI", "RACE_START"}:
                raise ValueError("a grid reference requires its own POST_QUALI snapshot")
            b = entries.copy()
            # Reference only: grid is POST_QUALI information, reported to bound
            # what a pre-weekend forecast could ever achieve.
            b["score"] = baseline_scores(panel, race_id, b, "grid")
            b["win_probability"] = np.nan
            b["model"] = "grid_reference_post_quali"
            b["race_id"] = race_id
            b["stage"] = "POST_QUALI"
            b["evaluation_scope"] = model_rows["evaluation_scope"].iloc[0]
            frames.append(b)

    if skipped:
        reasons = pd.DataFrame(skipped)["reason"].value_counts().to_dict()
        log.warning("skipped %d of %d races: %s", len(skipped), len(race_ids), reasons)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    predictions = pd.concat(frames, ignore_index=True, sort=False)
    per_race = []
    for model, g in predictions.groupby("model"):
        m = evaluate(g, score_col="score", prob_col="win_probability")
        m["model"] = model
        per_race.append(m)
    return predictions, pd.concat(per_race, ignore_index=True)


def leaderboard(per_race: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-race metrics into a model leaderboard.

    Top-1 carries its Wilson interval, because on 92 races the difference between
    two models is usually smaller than the interval around either of them.
    """
    rows = []
    for model, g in per_race.groupby("model"):
        summary = summarise(g, label=str(model))
        wins = int(round(summary.get("top1", 0.0) * len(g)))
        low, high = wilson_interval(wins, len(g))
        rows.append({**summary, "top1_ci_low": low, "top1_ci_high": high})
    lb = pd.DataFrame(rows)
    cols = [
        "model",
        "n_races",
        "top1",
        "top1_ci_low",
        "top1_ci_high",
        "winner_top3",
        "ndcg5",
        "spearman",
        "rank_mae",
        "winner_logloss",
        "winner_brier",
    ]
    return (
        lb[[c for c in cols if c in lb]].sort_values("top1", ascending=False).reset_index(drop=True)
    )
