"""Track-car compatibility: similarity kernel behaviour and safe fallbacks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pred.features.compatibility import (
    SIMILARITY_FEATURES,
    build_target_profile,
    circuit_similarity,
    similarity_weighted_observations,
)


@pytest.fixture
def profiles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "circuit_id": ["fast_permanent", "slow_street", "mid_street", "mid_permanent"],
            "avg_lap_speed_kmh": [260.0, 190.0, 212.0, 214.0],
            "lap_length_m": [5800.0, 4900.0, 5400.0, 5450.0],
            "field_spread_pct": [1.5, 1.4, 1.4, 1.45],
            "overtaking_difficulty": [0.15, 0.75, 0.55, 0.5],
            "is_street": [False, True, True, False],
        }
    )


def test_identical_circuit_scores_one(profiles):
    target = profiles.iloc[2]
    sim = circuit_similarity(target, profiles, bandwidth=1.0)
    assert sim["mid_street"] == pytest.approx(1.0, abs=1e-9)


def test_similarity_is_bounded_and_positive(profiles):
    sim = circuit_similarity(profiles.iloc[2], profiles, bandwidth=1.0)
    assert (sim > 0).all() and (sim <= 1.0 + 1e-9).all()


def test_dissimilar_circuit_scores_lower_than_similar_one(profiles):
    sim = circuit_similarity(profiles.iloc[2], profiles, bandwidth=1.0)
    assert sim["mid_permanent"] > sim["fast_permanent"]
    assert sim["slow_street"] > sim["fast_permanent"]


def test_street_bonus_favours_the_same_layout_family(profiles):
    target = profiles.iloc[2]
    with_bonus = circuit_similarity(target, profiles, bandwidth=1.0, street_bonus=1.0)
    without = circuit_similarity(target, profiles, bandwidth=1.0, street_bonus=0.0)
    assert with_bonus["slow_street"] / without["slow_street"] > 1.0


def test_larger_bandwidth_flattens_toward_the_pooled_model(profiles):
    target = profiles.iloc[2]
    narrow = circuit_similarity(target, profiles, bandwidth=0.4)
    wide = circuit_similarity(target, profiles, bandwidth=5.0)
    assert wide.std() < narrow.std()
    assert wide.min() > narrow.min()


def test_observation_weights_respect_the_floor(profiles):
    sim = pd.Series({"a": 1.0, "b": 0.0001})
    obs = pd.DataFrame({"circuit_id": ["a", "b", "unknown"]})
    w = similarity_weighted_observations(obs, sim, floor=0.2)
    assert w.iloc[0] == pytest.approx(1.0)
    assert w.iloc[1] == pytest.approx(0.2), "floor keeps dissimilar circuits contributing"
    assert w.iloc[2] >= 0.2, "unseen circuits fall back rather than vanishing"


def test_observation_weights_pass_through_without_a_circuit_column():
    obs = pd.DataFrame({"score": [1.0, 2.0]})
    w = similarity_weighted_observations(obs, pd.Series({"a": 1.0}))
    assert (w == 1.0).all()


def test_build_target_profile_uses_history_when_available(profiles):
    out = build_target_profile("mid_street", profiles)
    assert out["circuit_id"] == "mid_street"
    assert out["avg_lap_speed_kmh"] == pytest.approx(212.0)


def test_build_target_profile_synthesises_a_new_circuit(profiles):
    """The Madrid case: no history, so the published specification must be used."""
    out = build_target_profile(
        "madring",
        profiles,
        published_lap_length_m=5416.0,
        published_pole_time_s=92.0,
        is_street=True,
        overtaking_difficulty=0.567,
    )
    assert out["avg_lap_speed_kmh"] == pytest.approx(3.6 * 5416.0 / 92.0)
    assert out["is_street"] is True
    for col in SIMILARITY_FEATURES:
        assert np.isfinite(float(out[col]))


def test_new_circuit_without_a_specification_is_an_error(profiles):
    with pytest.raises(ValueError, match="no history"):
        build_target_profile("unknown_circuit", profiles)
