"""Latent strength model: scale conventions, identification, and leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pred.config import ModelConfig
from f1pred.features.pace import (
    build_observations,
    entry_strength,
    fit_strength,
    fit_strength_backfit,
    normal_scores,
    qualifying_pace_score,
    recency_weights,
)


def test_normal_scores_are_higher_is_better_and_symmetric():
    pos = pd.Series([1, 10, 20])
    n = pd.Series([20, 20, 20])
    s = normal_scores(pos, n)
    assert s.iloc[0] > s.iloc[1] > s.iloc[2]
    assert s.iloc[0] == pytest.approx(-s.iloc[2], abs=1e-9)
    assert abs(s.iloc[1]) < 0.15


def test_normal_scores_propagate_missing_positions():
    s = normal_scores(pd.Series([1.0, np.nan]), pd.Series([20, 20]))
    assert np.isfinite(s.iloc[0]) and np.isnan(s.iloc[1])


def test_qualifying_score_is_negative_gap_and_pole_is_zero():
    s = qualifying_pace_score(pd.Series([0.0, 0.75, 1.5]))
    assert s.iloc[0] == pytest.approx(0.0)
    assert s.iloc[1] == pytest.approx(-1.0)
    assert s.iloc[2] == pytest.approx(-2.0)


def test_qualifying_and_race_scores_share_a_scale():
    """Both must land on roughly the same range or the ridge fit is dominated by one."""
    race = normal_scores(pd.Series([1, 20]), pd.Series([20, 20])).abs().max()
    qual = qualifying_pace_score(pd.Series([1.5])).abs().max()
    assert 0.5 < qual / race < 2.0


def test_recency_weights_halve_at_the_half_life():
    w = recency_weights(pd.Series([10.0, 4.0, -2.0]), latest_index=10.0, half_life=6.0)
    assert w.iloc[0] == pytest.approx(1.0)
    assert w.iloc[1] == pytest.approx(0.5)
    assert w.iloc[2] == pytest.approx(0.25)


def test_recency_weights_never_exceed_one_for_future_rows():
    w = recency_weights(pd.Series([20.0]), latest_index=10.0, half_life=6.0)
    assert w.iloc[0] == pytest.approx(1.0)


def test_observations_exclude_retirements(synthetic_results, synthetic_qualifying):
    results = synthetic_results.copy()
    results.loc[results.index[:3], "classified"] = 0
    results["event_index"] = results.groupby("race_id", sort=False).ngroup().astype(float)
    qual = synthetic_qualifying.copy()
    qual["event_index"] = qual.groupby("race_id", sort=False).ngroup().astype(float)
    obs = build_observations(results, qual)
    race_obs = obs[obs["kind"] == "race"]
    assert len(race_obs) == int(results["classified"].sum())


def test_fit_recovers_known_team_and_driver_ordering(
    synthetic_results, synthetic_qualifying, synthetic_schedule
):
    """The fixture encodes team_a > team_b > team_c and d1 > d2 within team_a."""
    from f1pred.pipeline import build_panel

    panel = build_panel(synthetic_results, synthetic_qualifying, synthetic_schedule)
    est = fit_strength(
        panel.observations,
        as_of_index=panel.index_of("2025_10") + 1,
        target_regime="GROUND_EFFECT_2022",
    )
    teams = est.teams.set_index("constructor_id")["team_effect"]
    assert teams["team_a"] > teams["team_b"] > teams["team_c"]
    drivers = est.drivers.set_index("driver_id")["shrunk_effect"]
    assert drivers["d1"] > drivers["d2"]
    assert drivers["d3"] > drivers["d4"]


def test_fit_uses_only_observations_before_as_of_index(
    synthetic_results, synthetic_qualifying, synthetic_schedule
):
    """Corrupting only *future* races must leave the estimate untouched."""
    from f1pred.pipeline import build_panel

    panel = build_panel(synthetic_results, synthetic_qualifying, synthetic_schedule)
    cutoff = panel.index_of("2024_01")
    clean = fit_strength(panel.observations, as_of_index=cutoff, target_regime="GROUND_EFFECT_2022")

    poisoned = panel.observations.copy()
    future = poisoned["event_index"] >= cutoff
    poisoned.loc[future, "score"] = 999.0
    dirty = fit_strength(poisoned, as_of_index=cutoff, target_regime="GROUND_EFFECT_2022")

    pd.testing.assert_frame_equal(clean.teams, dirty.teams)
    pd.testing.assert_frame_equal(clean.drivers, dirty.drivers)


def test_empty_history_returns_neutral_estimate():
    obs = pd.DataFrame(
        columns=[
            "race_id",
            "season",
            "round",
            "driver_id",
            "constructor_id",
            "kind",
            "score",
            "regime",
            "event_index",
        ]
    )
    est = fit_strength(obs, as_of_index=0.0, target_regime="NEW_PU_AERO_2026")
    assert est.n_observations == 0
    assert est.teams.empty and est.drivers.empty


def test_rookie_gets_penalty_and_is_flagged():
    est = fit_strength(
        pd.DataFrame(
            {
                "race_id": ["r1"] * 4,
                "season": [2026] * 4,
                "round": [1] * 4,
                "driver_id": ["a", "b", "a", "b"],
                "constructor_id": ["t1", "t2", "t1", "t2"],
                "kind": ["race"] * 4,
                "score": [1.0, -1.0, 1.2, -1.2],
                "regime": ["NEW_PU_AERO_2026"] * 4,
                "event_index": [0.0, 0.0, 1.0, 1.0],
            }
        ),
        as_of_index=5.0,
        target_regime="NEW_PU_AERO_2026",
    )
    entries = pd.DataFrame(
        {"driver_id": ["a", "newcomer"], "constructor_id": ["t1", "t1"], "season": [2026, 2026]}
    )
    scored = entry_strength(est, entries, rookie_penalty=-0.25)
    assert not scored.loc[0, "is_rookie"]
    assert scored.loc[1, "is_rookie"]
    assert scored.loc[1, "driver_effect"] == pytest.approx(-0.25)


def test_backfit_matches_joint_fit_ordering(
    synthetic_results, synthetic_qualifying, synthetic_schedule
):
    """With equal half-lives the two estimators must agree on ordering."""
    from f1pred.pipeline import build_panel

    panel = build_panel(synthetic_results, synthetic_qualifying, synthetic_schedule)
    idx = panel.index_of("2025_10") + 1
    cfg = ModelConfig(recency_half_life_races=8.0)
    joint = fit_strength(
        panel.observations, as_of_index=idx, target_regime="GROUND_EFFECT_2022", config=cfg
    )
    bf = fit_strength_backfit(
        panel.observations,
        as_of_index=idx,
        target_regime="GROUND_EFFECT_2022",
        config=cfg,
        team_half_life=8.0,
        driver_half_life=8.0,
    )
    a = joint.teams.set_index("constructor_id")["team_effect"].rank()
    b = bf.teams.set_index("constructor_id")["team_effect"].rank()
    pd.testing.assert_series_equal(a, b, check_names=False)
