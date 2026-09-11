"""Circuit characterisation: measurement, transfer, and standardised similarity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pred.features.circuits import (
    RACE_DISTANCE_OVERRIDES_M,
    build_circuit_profiles,
    circuit_speed_profile,
    empirical_overtaking_difficulty,
    nearest_circuits,
    transfer_overtaking_difficulty,
)


def _races(circuit: str, n_races: int, shuffle: float, seed: int) -> pd.DataFrame:
    """Synthetic races where ``shuffle`` controls how much the order changes."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        grid = np.arange(1, 21)
        finish = np.argsort(np.argsort(grid + rng.normal(0, shuffle, 20))) + 1
        for g, f in zip(grid, finish, strict=True):
            rows.append(
                {
                    "circuit_id": circuit,
                    "race_id": f"{circuit}_{r}",
                    "season": 2020 + r % 5,
                    "grid": float(g),
                    "position_order": int(f),
                    "classified": 1,
                    "started": 1,
                    "dnf": 0,
                    "driver_id": f"d{g}",
                    "laps": 60,
                    "winner": int(f == 1),
                }
            )
    return pd.DataFrame(rows)


def test_processional_circuit_scores_harder_than_chaotic_one():
    df = pd.concat([_races("procession", 6, 0.4, 1), _races("chaos", 6, 8.0, 2)], ignore_index=True)
    out = empirical_overtaking_difficulty(df).set_index("circuit_id")
    assert (
        out.loc["procession", "overtaking_difficulty"] > out.loc["chaos", "overtaking_difficulty"]
    )
    assert out.loc["procession", "grid_finish_spearman"] > out.loc["chaos", "grid_finish_spearman"]
    assert (
        out.loc["procession", "mean_abs_position_change"]
        < out.loc["chaos", "mean_abs_position_change"]
    )


def test_difficulty_stays_in_the_unit_interval():
    df = pd.concat([_races(f"c{i}", 4, 0.5 + 2.0 * i, 10 + i) for i in range(5)], ignore_index=True)
    out = empirical_overtaking_difficulty(df)
    assert out["overtaking_difficulty"].between(0.0, 1.0).all()


def test_transfer_prefers_named_analogues_over_the_family_average():
    profiles = pd.DataFrame(
        {
            "circuit_id": ["baku", "miami", "monaco", "marina_bay", "monza"],
            "overtaking_difficulty": [0.55, 0.55, 0.95, 0.75, 0.15],
            "n_races": [6, 6, 8, 6, 8],
        }
    )
    value, source = transfer_overtaking_difficulty(
        profiles, is_street=True, reference_circuits=["baku", "miami"]
    )
    assert value == pytest.approx(0.55)
    assert "analogues" in source and "baku" in source


def test_transfer_falls_back_to_the_layout_family():
    profiles = pd.DataFrame(
        {
            "circuit_id": ["baku", "monaco", "monza"],
            "overtaking_difficulty": [0.55, 0.95, 0.15],
            "n_races": [6, 8, 8],
        }
    )
    value, source = transfer_overtaking_difficulty(profiles, is_street=True)
    assert "family=street" in source
    # Must reflect street circuits only, not be dragged down by Monza.
    assert value == pytest.approx(0.75)


def test_nearest_circuits_standardises_before_measuring():
    """Without standardisation a large-magnitude column would dominate entirely."""
    profiles = pd.DataFrame(
        {
            "circuit_id": ["a", "b", "c"],
            "avg_lap_speed_kmh": [210.0, 260.0, 190.0],
            "lap_length_m": [5400.0, 5400.0, 5400.0],
        }
    )
    target = pd.Series({"avg_lap_speed_kmh": 212.0, "lap_length_m": 5400.0})
    out = nearest_circuits(target, profiles, ["avg_lap_speed_kmh", "lap_length_m"], n=1)
    assert out["circuit_id"].iloc[0] == "a"


def test_nearest_circuits_respects_the_restriction_set():
    profiles = pd.DataFrame(
        {
            "circuit_id": ["a", "b"],
            "avg_lap_speed_kmh": [210.0, 250.0],
            "lap_length_m": [5400.0, 5400.0],
        }
    )
    target = pd.Series({"avg_lap_speed_kmh": 211.0, "lap_length_m": 5400.0})
    out = nearest_circuits(
        target, profiles, ["avg_lap_speed_kmh", "lap_length_m"], n=2, restrict_to={"b"}
    )
    assert list(out["circuit_id"]) == ["b"]


def test_monaco_race_distance_override_is_registered():
    """Without it Monaco's recovered lap length is inflated by roughly 17%."""
    assert RACE_DISTANCE_OVERRIDES_M["monaco"] == pytest.approx(260_000.0)


def test_speed_profile_recovers_lap_length_from_lap_count():
    results = pd.DataFrame(
        {
            "circuit_id": ["test"] * 2,
            "race_id": ["r1", "r2"],
            "season": [2023, 2024],
            "winner": [1, 1],
            "laps": [53, 53],
            "started": [1, 1],
            "classified": [1, 1],
            "dnf": [0, 0],
            "driver_id": ["d1", "d1"],
            "grid": [1.0, 1.0],
            "position_order": [1, 1],
        }
    )
    qual = pd.DataFrame(
        {
            "circuit_id": ["test"] * 2,
            "race_id": ["r1", "r2"],
            "season": [2023, 2024],
            "qual_best_s": [90.0, 90.0],
            "qual_gap_pct": [0.0, 0.0],
        }
    )
    profile = circuit_speed_profile(results, qual).set_index("circuit_id")
    assert profile.loc["test", "lap_length_m"] == pytest.approx(305_000 / 53)
    assert profile.loc["test", "avg_lap_speed_kmh"] == pytest.approx(3.6 * (305_000 / 53) / 90.0)


def test_build_profiles_joins_overtaking_and_attrition():
    df = pd.concat([_races("a", 4, 1.0, 3), _races("b", 4, 5.0, 4)], ignore_index=True)
    out = build_circuit_profiles(df)
    for col in ("circuit_id", "overtaking_difficulty", "dnf_multiplier", "is_street"):
        assert col in out.columns


def test_measured_overtaking_normalises_for_race_length():
    """Monaco's 78 laps must not look like more overtaking than Spa's 44."""
    from f1pred.features.circuits import measured_overtaking_rate

    overtakes = pd.DataFrame(
        {
            "race_id": ["a"] * 100 + ["b"] * 100,
            "circuit_id": ["long_race"] * 100 + ["short_race"] * 100,
        }
    )
    results = pd.DataFrame(
        {
            "race_id": ["a"] * 20 + ["b"] * 20,
            "driver_id": [f"d{i}" for i in range(20)] * 2,
            "classified": [1] * 40,
            "laps": [78] * 20 + [44] * 20,
        }
    )
    rate = measured_overtaking_rate(overtakes, results, min_races=1).set_index("circuit_id")
    # Same raw count, so the longer race must show the lower rate per lap.
    assert (
        rate.loc["long_race", "overtakes_per_100_laps"]
        < (rate.loc["short_race", "overtakes_per_100_laps"])
    )
    assert rate.loc["short_race", "overtakes_per_100_laps"] == pytest.approx(100 * 100 / (44 * 20))


def test_pit_lane_loss_discards_repairs_and_timing_artefacts():
    from f1pred.features.circuits import pit_lane_loss

    pit = pd.DataFrame(
        {
            "circuit_id": ["x"] * 12,
            # Two implausible values that must not move the median.
            "pit_duration": [22.0] * 10 + [3.0, 400.0],
        }
    )
    out = pit_lane_loss(pit, min_stops=5).set_index("circuit_id")
    assert out.loc["x", "pit_lane_seconds"] == pytest.approx(22.0)
    assert out.loc["x", "n_stops"] == 10


def test_empty_overtake_input_returns_the_declared_columns():
    from f1pred.features.circuits import measured_overtaking_rate

    out = measured_overtaking_rate(pd.DataFrame(), pd.DataFrame())
    assert list(out.columns) == ["circuit_id", "overtakes_per_100_laps", "n_races"]
