"""End-to-end integration: the forecast path must be identical to the backtest path
and must refuse to run when a stage would leak."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pred.evaluation.backtest import race_entries, walk_forward
from f1pred.evaluation.metrics import evaluate
from f1pred.leakage import LeakageError
from f1pred.pipeline import build_panel, forecast_race, resolve_circuit
from f1pred.sim.race import SimulationSettings


@pytest.fixture
def panel(synthetic_results, synthetic_qualifying, synthetic_schedule):
    return build_panel(synthetic_results, synthetic_qualifying, synthetic_schedule)


def test_panel_indexes_races_chronologically(panel):
    first = panel.index_of("2022_01")
    later = panel.index_of("2025_10")
    assert later > first
    with pytest.raises(ValueError, match="not on the calendar"):
        panel.index_of("9999_01")


def test_forecast_produces_coherent_probabilities(panel):
    entries = race_entries(panel, "2025_05")
    fc = forecast_race(
        panel,
        "2025_05",
        entries,
        stage="PRE_WEEKEND",
        allow_retrospective=True,
        settings=SimulationSettings(n_sims=2000),
    )
    t = fc.table
    assert t["win_probability"].sum() == pytest.approx(1.0, abs=1e-9)
    assert set(
        ["win_probability", "podium_probability", "dnf_probability", "expected_finish"]
    ) <= set(t.columns)
    assert len(t) == len(entries)


def test_forecast_rejects_grid_before_qualifying(panel):
    entries = race_entries(panel, "2025_05")
    entries["grid"] = np.arange(1, len(entries) + 1)
    with pytest.raises(LeakageError, match="grid"):
        forecast_race(
            panel,
            "2025_05",
            entries,
            stage="PRE_WEEKEND",
            grid_col="grid",
            allow_retrospective=True,
            settings=SimulationSettings(n_sims=200),
        )


def test_forecast_accepts_grid_after_qualifying(panel):
    entries = race_entries(panel, "2025_05")
    entries["grid"] = np.arange(1, len(entries) + 1)
    fc = forecast_race(
        panel,
        "2025_05",
        entries,
        stage="POST_QUALI",
        grid_col="grid",
        allow_retrospective=True,
        settings=SimulationSettings(n_sims=1000),
    )
    assert fc.diagnostics["grid_weight"] > 0


def test_forecast_is_unchanged_by_corrupting_the_target_race(
    panel, synthetic_results, synthetic_qualifying, synthetic_schedule
):
    """The strongest leakage test: destroy the target race's own outcome and the
    forecast must not move at all."""
    entries = race_entries(panel, "2024_05")
    clean = forecast_race(
        panel,
        "2024_05",
        entries,
        stage="PRE_WEEKEND",
        allow_retrospective=True,
        settings=SimulationSettings(n_sims=3000),
    )

    corrupted = synthetic_results.copy()
    mask = corrupted["race_id"] == "2024_05"
    corrupted.loc[mask, "position_order"] = corrupted.loc[mask, "position_order"].to_numpy()[::-1]
    corrupted.loc[mask, "winner"] = 0
    corrupt_panel = build_panel(corrupted, synthetic_qualifying, synthetic_schedule)
    dirty = forecast_race(
        corrupt_panel,
        "2024_05",
        entries,
        stage="PRE_WEEKEND",
        allow_retrospective=True,
        settings=SimulationSettings(n_sims=3000),
    )

    np.testing.assert_allclose(
        clean.table.sort_values("driver_id")["win_probability"].to_numpy(),
        dirty.table.sort_values("driver_id")["win_probability"].to_numpy(),
        atol=1e-12,
    )


def test_race_entries_are_not_ordered_by_outcome(panel):
    """Row order must never encode the answer; ties downstream would read it."""
    entries = race_entries(panel, "2024_05")
    assert list(entries["driver_id"]) == sorted(entries["driver_id"])


def test_evaluation_is_invariant_to_input_row_order(panel):
    """A metric that changes when rows are shuffled is reading the row order."""
    entries = race_entries(panel, "2024_05")
    fc = forecast_race(
        panel,
        "2024_05",
        entries,
        stage="PRE_WEEKEND",
        allow_retrospective=True,
        settings=SimulationSettings(n_sims=2000),
    )
    table = fc.table
    a = evaluate(table, score_col="score", prob_col="win_probability")
    b = evaluate(
        table.sample(frac=1.0, random_state=0), score_col="score", prob_col="win_probability"
    )
    pd.testing.assert_frame_equal(
        a.reset_index(drop=True), b.reset_index(drop=True), check_like=True
    )


def test_tied_scores_do_not_leak_the_finishing_order():
    """With every score identical the metric must be indifferent to the truth."""
    n = 20
    race = pd.DataFrame(
        {
            "race_id": ["r"] * n,
            "driver_id": [f"d{i:02d}" for i in range(n)],
            "position_order": np.arange(1, n + 1),
            "classified": 1,
            "score": 0.0,
        }
    )
    forward = evaluate(race, score_col="score", prob_col=None)
    backward = evaluate(race.iloc[::-1], score_col="score", prob_col=None)
    assert forward["spearman"].iloc[0] == pytest.approx(backward["spearman"].iloc[0])


def test_walk_forward_runs_and_scores_every_model(panel):
    ids = ["2025_05", "2025_06", "2025_07"]
    predictions, per_race = walk_forward(panel, ids, n_sims=800, allow_retrospective=True)
    assert not predictions.empty
    models = set(per_race["model"])
    assert {"strength_mc", "gp_points", "form5", "last_race"} <= models
    assert per_race["top1"].between(0, 1).all()


def test_resolve_circuit_falls_back_for_an_unknown_venue(panel):
    ctx = resolve_circuit(panel, "2025_10", "a_circuit_never_visited")
    assert 0.0 < ctx.overtaking_difficulty < 1.0
    assert ctx.source.startswith("transfer")
