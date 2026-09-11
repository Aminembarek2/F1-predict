"""Leakage guards. These are the tests that matter most: a silent leak invalidates
every other number the project reports."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pred.leakage import (
    FEATURE_CATALOG,
    LeakageError,
    assert_observed_before,
    assert_stage_allowed,
    earliest_stage,
    is_allowed,
    prediction_cutoff,
    shifted_expanding,
)


def test_grid_is_illegal_before_qualifying():
    assert not is_allowed("grid", "PRE_WEEKEND")
    assert not is_allowed("grid", "POST_FP3")
    assert is_allowed("grid", "POST_QUALI")


def test_race_outcomes_are_never_legal():
    for family in ("race_result", "race_laps", "race_weather_observed", "race_pitstops"):
        for stage in ("PRE_WEEKEND", "POST_FP3", "POST_QUALI", "RACE_START"):
            assert not is_allowed(family, stage), f"{family} leaked at {stage}"


def test_unregistered_family_fails_closed():
    """An unknown feature must be rejected, not silently permitted."""
    assert earliest_stage("some_new_idea_nobody_registered") == "POST_RACE"
    assert not is_allowed("some_new_idea_nobody_registered", "RACE_START")


def test_assert_stage_allowed_raises_with_detail():
    with pytest.raises(LeakageError, match="grid"):
        assert_stage_allowed(["historical_results", "grid"], "PRE_WEEKEND")
    assert assert_stage_allowed(["historical_results", "reliability"], "PRE_WEEKEND") is None


def test_practice_stages_are_ordered():
    assert not is_allowed("practice_fp2", "POST_FP1")
    assert is_allowed("practice_fp1", "POST_FP1")
    assert is_allowed("practice_fp2", "POST_FP3")


def test_every_catalog_entry_declares_a_known_stage():
    valid = {
        "PRE_WEEKEND",
        "POST_FP1",
        "POST_FP2",
        "POST_FP3",
        "POST_QUALI",
        "RACE_START",
        "POST_RACE",
    }
    for family, spec in FEATURE_CATALOG.items():
        assert spec.earliest_stage in valid, family
        assert spec.source and spec.formula, family


def test_timestamp_guard_rejects_future_rows():
    df = pd.DataFrame({"observed": pd.to_datetime(["2026-09-10", "2026-09-14"], utc=True)})
    assert_observed_before(df.iloc[:1], "observed", "2026-09-11")
    with pytest.raises(LeakageError, match="1 rows"):
        assert_observed_before(df, "observed", "2026-09-11")


def test_shifted_expanding_never_sees_current_row():
    df = pd.DataFrame({"g": ["a"] * 4, "order": [1, 2, 3, 4], "v": [10.0, 20.0, 30.0, 40.0]})
    out = shifted_expanding(df, ["g"], "v", "order")
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(10.0)
    assert out.iloc[2] == pytest.approx(15.0)
    assert out.iloc[3] == pytest.approx(20.0)


def test_shifted_expanding_respects_groups():
    df = pd.DataFrame(
        {"g": ["a", "a", "b", "b"], "order": [1, 2, 1, 2], "v": [10.0, 20.0, 100.0, 200.0]}
    )
    out = shifted_expanding(df, ["g"], "v", "order")
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[2])
    assert out.iloc[1] == pytest.approx(10.0)
    assert out.iloc[3] == pytest.approx(100.0)


def test_shifted_expanding_window_is_bounded():
    df = pd.DataFrame({"g": ["a"] * 5, "order": range(1, 6), "v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = shifted_expanding(df, ["g"], "v", "order", window=2)
    # Last value averages rows 3 and 4 only, never row 5.
    assert out.iloc[4] == pytest.approx(3.5)


def test_prediction_cutoffs_are_monotonic_and_before_the_race():
    start = pd.Timestamp("2026-09-13 13:00", tz="UTC")
    stages = ["PRE_WEEKEND", "POST_FP1", "POST_FP2", "POST_FP3", "POST_QUALI"]
    cutoffs = [prediction_cutoff(start, s) for s in stages]
    assert cutoffs == sorted(cutoffs)
    assert all(c < start for c in cutoffs)
    assert prediction_cutoff(start, "RACE_START") == start
