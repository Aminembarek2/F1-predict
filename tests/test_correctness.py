"""Regressions for the defects documented in the repository audit."""

import numpy as np
import pandas as pd
import pytest

from f1pred.evaluation.backtest import race_entries, walk_forward
from f1pred.evaluation.metrics import paired_bootstrap, race_metrics
from f1pred.forecast import expected_entry_list, forecast_upcoming
from f1pred.leakage import LeakageError, assert_observed_before, shifted_expanding
from f1pred.pipeline import baseline_scores, build_panel, forecast_race
from f1pred.sim.race import SimulationSettings, conditional_dnf_rates


@pytest.fixture
def panel(synthetic_results, synthetic_qualifying, synthetic_schedule):
    return build_panel(synthetic_results, synthetic_qualifying, synthetic_schedule)


def test_ingestion_gap_cannot_admit_target_observations(
    synthetic_results, synthetic_qualifying, synthetic_schedule
):
    panel = build_panel(
        synthetic_results.query("race_id != '2022_01'"),
        synthetic_qualifying.query("race_id != '2022_01'"),
        synthetic_schedule,
    )
    cutoff = panel.index_of("2024_05")
    past = panel.observations.loc[panel.observations.event_index < cutoff]
    assert "2024_05" not in set(past.race_id)
    assert (
        panel.observations.loc[panel.observations.race_id.eq("2024_05"), "event_index"] == cutoff
    ).all()


def test_unmapped_race_is_rejected(synthetic_results, synthetic_qualifying, synthetic_schedule):
    with pytest.raises(ValueError, match="absent from the calendar"):
        build_panel(synthetic_results, synthetic_qualifying, synthetic_schedule.iloc[1:])


@pytest.mark.parametrize("value", [None, "nonsense", "2026-09-12"])
def test_timestamps_fail_closed(value):
    with pytest.raises(LeakageError):
        assert_observed_before(pd.DataFrame({"observed": [value]}), "observed", "2026-09-11")


def test_same_event_teammates_cannot_leak_through_shift():
    data = pd.DataFrame({"team": ["a", "a"], "event": [1, 1], "value": [10, 20]})
    with pytest.raises(LeakageError, match="one row"):
        shifted_expanding(data, ["team"], "value", "event")


def test_strict_forecast_requires_cutoff_and_source(panel):
    entries = race_entries(panel, "2025_05")
    with pytest.raises(LeakageError, match="cutoff"):
        forecast_race(panel, "2025_05", entries)
    with pytest.raises(LeakageError, match="observed_at"):
        forecast_race(panel, "2025_05", entries, cutoff="2025-04-28")


def test_timestamp_gate_runs_in_forecast(panel):
    """History published after the cutoff must stop the forecast."""
    entries = race_entries(panel, "2025_05")
    entries["observed_at"], entries["source"] = "2020-01-01", "fixture"
    # A cutoff before the earlier rounds were published makes their results
    # unavailable, so the forecast must refuse rather than quietly use them.
    with pytest.raises(LeakageError, match="at/after"):
        forecast_race(panel, "2025_05", entries, cutoff="2020-01-02")


def test_download_time_is_provenance_and_not_a_leak(
    synthetic_results, synthetic_qualifying, synthetic_schedule
):
    """Re-fetching an old race today does not make its outcome future information.

    The gate is publication time. Treating the download timestamp as availability
    would block every live forecast, since that timestamp is always 'now'.
    """
    results, qualifying = synthetic_results.copy(), synthetic_qualifying.copy()
    results["observed_at"] = "2030-01-01"
    qualifying["observed_at"] = "2030-01-01"
    panel = build_panel(results, qualifying, synthetic_schedule)
    entries = race_entries(panel, "2025_05")
    entries["observed_at"], entries["source"] = "2025-04-01", "fixture"
    forecast = forecast_race(panel, "2025_05", entries, cutoff="2025-04-28")
    assert forecast.table["evaluation_scope"].eq("reconstructed_snapshot").all()


def test_live_and_backtest_use_the_same_grid_and_seed(panel):
    entries = race_entries(panel, "2025_05")
    live = forecast_upcoming(
        panel,
        season=2025,
        round_no=5,
        entries=entries,
        stage="POST_QUALI",
        n_sims=1000,
        allow_retrospective=True,
    )
    back, _ = walk_forward(
        panel, ["2025_05"], stage="POST_QUALI", n_sims=1000, allow_retrospective=True
    )
    assert "pole_probability" not in live.table
    pd.testing.assert_series_equal(
        live.table.set_index("driver_id").win_probability.sort_index(),
        back.loc[back.model.eq("strength_mc")].set_index("driver_id").win_probability.sort_index(),
    )


def test_postquali_missing_grid_is_an_error(panel):
    entries = race_entries(panel, "2025_05").drop(columns="grid")
    with pytest.raises(ValueError, match="complete grid"):
        forecast_upcoming(
            panel,
            season=2025,
            round_no=5,
            entries=entries,
            stage="POST_QUALI",
            n_sims=10,
            allow_retrospective=True,
        )


def test_pitlane_starter_is_retained(panel):
    mask = panel.results.race_id.eq("2025_05") & panel.results.driver_id.eq("d6")
    panel.results.loc[mask, "grid"] = 0
    pred, _ = walk_forward(
        panel, ["2025_05"], stage="POST_QUALI", n_sims=500, allow_retrospective=True
    )
    assert "d6" in set(pred.loc[pred.model.eq("strength_mc"), "driver_id"])


def test_season_opener_requires_a_lineup(panel):
    with pytest.raises(ValueError, match="season openers"):
        expected_entry_list(panel, 2025, 1)


@pytest.mark.parametrize("chaos_probability", [0, 0.22, 1])
def test_chaos_preserves_marginals_including_high_risks(chaos_probability):
    p = np.array([0, 0.1, 0.5, 0.8, 1.0])
    calm, chaos = conditional_dnf_rates(p, chaos_probability, 2.1)
    np.testing.assert_allclose((1 - chaos_probability) * calm + chaos_probability * chaos, p)
    assert np.all(chaos >= calm)


def test_missing_probability_is_not_silently_uniform(panel):
    entries = race_entries(panel, "2025_05")
    entries["score"], entries["win_probability"] = -entries.grid, np.nan
    result = race_metrics(entries, "score", "win_probability")
    assert np.isnan(result["winner_logloss"])


def test_official_standings_are_required_instead_of_gp_points(panel):
    with pytest.raises(ValueError, match="official standings"):
        baseline_scores(panel, "2025_05", race_entries(panel, "2025_05"), "championship")


def test_lower_logloss_has_favourable_bootstrap_mass():
    a = pd.DataFrame({"race_id": ["a", "b"], "winner_logloss": [2.0, 3.0]})
    b = a.assign(winner_logloss=[1.0, 2.0])
    result = paired_bootstrap(a, b, "winner_logloss")
    assert result["delta"] == -1 and result["p_better"] == 1


def test_display_rank_matches_evaluated_score(panel):
    pred, _ = walk_forward(panel, ["2025_05"], n_sims=1000, allow_retrospective=True)
    model = pred.loc[pred.model.eq("strength_mc")].sort_values("predicted_rank")
    assert model.score.is_monotonic_decreasing


def test_invalid_simulation_budget_is_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        SimulationSettings(n_sims=0)


def test_probability_report_scores_every_published_probability():
    """A low winner ECE must not stand in for podium, top-10 or DNF calibration."""
    from f1pred.evaluation.metrics import probability_report

    predictions = pd.DataFrame(
        {
            "race_id": ["r1"] * 4,
            "win_probability": [0.7, 0.2, 0.1, 0.0],
            "podium_probability": [0.9, 0.8, 0.7, 0.6],
            "top10_probability": [1.0, 1.0, 1.0, 1.0],
            "dnf_probability": [0.1, 0.1, 0.1, 0.1],
            "position_order": [1, 2, 3, 4],
            "classified": [1, 1, 1, 1],
            "finish_p10": [1, 1, 1, 1],
            "finish_p90": [3, 3, 3, 3],
        }
    )
    report = probability_report(predictions).set_index("outcome")
    assert set(report.index) == {"winner", "podium", "top10", "dnf", "finish_inside_p10_p90"}
    assert report.loc["podium", "observed_rate"] == pytest.approx(0.75)
    assert report.loc["top10", "observed_rate"] == pytest.approx(1.0)
    assert report.loc["dnf", "observed_rate"] == pytest.approx(0.0)
    # The fourth car finished outside its own published P10-P90 band.
    assert report.loc["finish_inside_p10_p90", "observed_rate"] == pytest.approx(0.75)


def test_paired_table_compares_every_model_against_the_incumbent():
    from f1pred.evaluation.metrics import paired_table

    rng = np.random.default_rng(0)
    races = [f"r{i}" for i in range(20)]
    per_race = pd.concat(
        [
            pd.DataFrame({"race_id": races, "model": "strength_mc", "ndcg5": rng.random(20)}),
            pd.DataFrame({"race_id": races, "model": "rival", "ndcg5": rng.random(20)}),
        ]
    )
    out = paired_table(per_race, incumbent="strength_mc", metrics=("ndcg5",))
    assert list(out["model"]) == ["rival"]
    assert out["n_races"].iloc[0] == 20
    assert out["ci_low"].iloc[0] <= out["delta"].iloc[0] <= out["ci_high"].iloc[0]


def test_a_named_driver_who_does_not_start_is_scored_behind_the_field():
    """The entry list is itself a forecast, and being wrong about it must cost."""
    from f1pred.evaluation.backtest import align_snapshot_to_result

    entries = pd.DataFrame({"driver_id": ["a", "b", "absent"]})
    truth = pd.DataFrame(
        {
            "driver_id": ["a", "b", "unforeseen"],
            "position_order": [2, 3, 1],
            "classified": [1, 1, 1],
            "dnf": [0, 0, 0],
            "winner": [0, 0, 1],
        }
    )
    aligned = align_snapshot_to_result(entries, truth)
    # The winner was never named, so no ranking of this entry list can be right.
    assert not aligned.scoreable
    assert aligned.did_not_start == 1 and aligned.unforeseen == 1

    positions = aligned.truth.set_index("driver_id")["position_order"]
    assert positions["a"] < positions["b"] < positions["absent"]
    assert aligned.truth.set_index("driver_id").loc["absent", "classified"] == 0
    assert "unforeseen" not in set(aligned.truth["driver_id"])


def test_a_named_winner_makes_the_race_scoreable():
    from f1pred.evaluation.backtest import align_snapshot_to_result

    entries = pd.DataFrame({"driver_id": ["a", "b"]})
    truth = pd.DataFrame(
        {
            "driver_id": ["a", "b"],
            "position_order": [1, 2],
            "classified": [1, 1],
            "dnf": [0, 0],
            "winner": [1, 0],
        }
    )
    aligned = align_snapshot_to_result(entries, truth)
    assert aligned.scoreable and aligned.did_not_start == 0 and aligned.unforeseen == 0


def test_season_opener_baseline_uses_the_previous_final_standings(
    synthetic_results, synthetic_qualifying, synthetic_schedule
):
    """A flat zero at an opener makes the baseline degenerate where it is easiest
    to beat, which flatters anything measured against it."""
    standings = pd.DataFrame(
        {
            "season": [2024] * 6,
            "after_round": [10] * 6,
            "driver_id": ["d6", "d5", "d4", "d3", "d2", "d1"],
            "championship_position": [1, 2, 3, 4, 5, 6],
        }
    )
    panel = build_panel(synthetic_results, synthetic_qualifying, synthetic_schedule, standings)
    entries = race_entries(panel, "2025_01")
    scores = baseline_scores(panel, "2025_01", entries, "championship").to_numpy()
    assert len(set(scores)) > 1, "an opener must not leave every driver tied"
    ordered = entries.assign(score=scores).sort_values("score", ascending=False)
    assert list(ordered["driver_id"]) == ["d6", "d5", "d4", "d3", "d2", "d1"]


def test_sustained_rate_limit_is_enforced_not_just_spacing(tmp_path, monkeypatch):
    """A minimum spacing cannot honour a published per-minute quota on its own.

    Three requests a second satisfies any 0.34 s spacing rule and still reaches
    ninety a minute, so the window has to be enforced separately.
    """
    import f1pred.http_cache as http_cache

    clock = {"now": 0.0}
    slept: list[float] = []
    monkeypatch.setattr(http_cache.time, "monotonic", lambda: clock["now"])

    def fake_sleep(seconds):
        slept.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(http_cache.time, "sleep", fake_sleep)

    session = http_cache.CachedSession("probe", 0.34, cache_dir=tmp_path, burst=(30, 60.0))
    for _ in range(30):
        session._throttle()
    assert sum(slept) < 60.0, "the first thirty requests must not wait out a whole window"

    slept.clear()
    session._throttle()
    assert slept and sum(slept) > 0, "the thirty-first request must wait for the window"


def test_two_practice_sources_concatenate_without_a_mixed_column():
    """One source returns timestamps as datetimes, the other as ISO strings.

    Left unnormalised that difference survives to the parquet writer, which then
    refuses the column, so the contract is enforced at the source boundary.
    """
    from f1pred.sources import PRACTICE_REQUIRED, normalise_practice

    base = {
        "race_id": ["2024_01"],
        "season": [2024],
        "stage": ["POST_FP2"],
        "driver_id": ["x"],
        "lap_duration": ["92.5"],
        "tyre_age": [3],
        "compound": ["soft"],
        "stint_number": [1],
        "is_pit_out_lap": [0],
        "interrupted": [0],
        "rainfall": [0],
    }
    as_strings = pd.DataFrame(
        {**base, **{c: ["2024-02-29T15:40:28+00:00"] for c in PRACTICE_REQUIRED[-4:]}}
    )
    as_datetimes = pd.DataFrame(
        {**base, **{c: [pd.Timestamp("2024-02-29T15:40:28Z")] for c in PRACTICE_REQUIRED[-4:]}}
    )
    merged = pd.concat(
        [normalise_practice(as_strings), normalise_practice(as_datetimes)], ignore_index=True
    )
    assert str(merged["observed_at"].dtype) == "datetime64[ns, UTC]"
    assert merged["lap_duration"].dtype.kind == "f"
    assert merged["compound"].tolist() == ["SOFT", "SOFT"]
    assert merged["is_pit_out_lap"].dtype == bool


def test_a_practice_source_cannot_omit_a_contract_column():
    from f1pred.sources import normalise_practice

    with pytest.raises(ValueError, match="missing"):
        normalise_practice(pd.DataFrame({"race_id": ["2024_01"], "driver_id": ["x"]}))


def test_verified_publication_timestamps_override_date_proxy(
    synthetic_results, synthetic_qualifying, synthetic_schedule
):
    results, qualifying = synthetic_results.copy(), synthetic_qualifying.copy()
    results["published_at"] = "2030-01-01"
    qualifying["published_at"] = "2030-01-01"
    panel = build_panel(results, qualifying, synthetic_schedule)
    entries = race_entries(panel, "2025_05")
    entries["observed_at"], entries["source"] = "2025-04-01", "fixture"
    with pytest.raises(LeakageError, match="at/after"):
        forecast_race(panel, "2025_05", entries, cutoff="2025-04-28")
