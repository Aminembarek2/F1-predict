import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from f1pred.legacy.data_sources import (
    openf1_url,
)
from f1pred.legacy.features import (
    aero_commitment_index,
    backward_asof_weather,
    braking_efficiency,
    corner_metrics,
    cosine_compatibility,
    curvature_from_xy,
    curvature_indices,
    dirty_air_penalty,
    exit_amplification,
    fit_overtake_model,
    fuel_correct,
    nearest_tracks,
    pit_loss,
    practice_pace,
    reliability_features,
    resample_telemetry,
    segment_corners,
    shrink_mean,
    straight_time_gain,
    teammate_adjusted_decomposition,
    temperature_sensitivity,
    track_demand_vector,
    track_evolution_correct,
    traction_index,
    traffic_class,
    tyre_degradation,
    tyre_warmup_laps,
    undercut_power,
    wet_skill_residual,
    wind_components,
)
from f1pred.legacy.leakage import (
    allowed,
    assert_no_future_features,
    assert_timestamp_cutoff,
)
from f1pred.legacy.modeling import (
    monte_carlo_race,
)


def test_01_curvature_circle():
    th = np.linspace(0, 2 * np.pi, 500)
    r = 100.0
    k = curvature_from_xy(r * np.cos(th), r * np.sin(th))
    assert abs(np.nanmedian(k[10:-10]) - 1 / r) < 5e-4


def test_02_curvature_indices():
    x = curvature_indices([0.01, 0.02, 0.03, 0.04])
    assert x["curvature_severity"] > x["curvature_exposure"] > 0


def test_03_segment_corners():
    tel = pd.DataFrame({"distance": np.arange(0, 1001, 10), "speed": 200})
    mk = pd.DataFrame({"corner": [1, 2], "distance": [200, 700]})
    out = segment_corners(tel, mk, 50, 50)
    assert set(out.corner) == {1, 2} and out.relative_distance.abs().max() <= 50


def test_04_braking_efficiency():
    a = braking_efficiency(300, 100, 100)
    assert 20 < a < 40


def test_05_traction_index():
    a = traction_index(100, 172, 2.0)
    assert abs(a - 10) < 0.1


def test_06_exit_amplification():
    assert exit_amplification(3.6, 800) == pytest.approx(800.0)


def test_07_straight_time_gain():
    assert straight_time_gain(200, 190, 1000) > 0


def test_08_aero_commitment():
    assert aero_commitment_index(250, 0.01, 100) > aero_commitment_index(200, 0.01, 70)


def test_09_wind_projection():
    h, c = wind_components(10, 0, 0)
    assert h == pytest.approx(10) and c == pytest.approx(0, abs=1e-8)
    h, c = wind_components(10, 90, 0)
    assert abs(h) < 1e-7 and c == pytest.approx(10)


def test_10_corner_metrics():
    d = np.linspace(0, 300, 61)
    speed = np.r_[np.linspace(250, 100, 31), np.linspace(100, 220, 30)]
    tel = pd.DataFrame(
        {
            "distance": d,
            "speed": speed,
            "brake": np.r_[np.zeros(10), np.ones(21), np.zeros(30)],
            "throttle": 80,
            "curvature": 0.01,
        }
    )
    m = corner_metrics(tel, 500)
    assert (
        m["entry_speed_kmh"] >= 240 and m["apex_speed_kmh"] <= 105 and m["traction_index_ms2"] > 0
    )


def test_11_track_demand_vector():
    c = pd.DataFrame(
        {
            "apex_speed_kmh": [100, 180, 240, 250],
            "braking_efficiency_ms2": [10, 8, 4, 3],
            "traction_index_ms2": [5, 4, 2, 2],
            "aero_commitment": [0.2, 0.5, 1.2, 1.4],
            "exit_amplification": [10, 20, 30, 40],
        }
    )
    v = track_demand_vector(c)
    assert v.low_speed_share == pytest.approx(0.25) and v.high_speed_share == pytest.approx(0.5)


def test_12_shrink_mean():
    assert shrink_mean([10], 0, strength=1) == pytest.approx(5)


def test_13_cosine_compatibility():
    assert cosine_compatibility([1, 0], [1, 0]) == pytest.approx(1)
    assert abs(cosine_compatibility([1, 0], [0, 1])) < 1e-9


def test_14_nearest_tracks():
    hist = pd.DataFrame({"a": [1, 0, 0.9], "b": [0, 1, 0.1]})
    n = nearest_tracks(pd.Series({"a": 1, "b": 0}), hist, ["a", "b"], 2)
    assert n.iloc[0]["index"] == 0


def test_15_teammate_adjusted_decomposition():
    rows = []
    for car, base in [("A", 1), ("B", -1)]:
        for drv, skill in [(car + "1", 0.5), (car + "2", -0.5)]:
            for _ in range(10):
                rows.append({"car": car, "driver": drv, "performance": base + skill})
    dr, ca = teammate_adjusted_decomposition(pd.DataFrame(rows), alpha=0.1)
    assert len(dr) == 4 and len(ca) == 2


def test_16_temperature_sensitivity():
    t = np.linspace(20, 50, 50)
    d = pd.DataFrame(
        {
            "track_temp": t,
            "pace_residual": 0.03 * t + np.random.default_rng(1).normal(0, 0.01, len(t)),
        }
    )
    assert abs(temperature_sensitivity(d) - 0.03) < 0.01


def test_17_wet_skill_residual():
    d = pd.DataFrame(
        {
            "driver": ["A", "B", "A", "B"],
            "car": ["X"] * 4,
            "pace_residual": [-0.2, 0.2, -0.1, 0.1],
            "wet": [1, 1, 1, 1],
        }
    )
    s = wet_skill_residual(d)
    assert s["A"] > s["B"]


def test_18_tyre_degradation_with_temp_and_missing():
    age = np.arange(1, 41)
    temp = np.random.default_rng(3).uniform(25, 45, 40)
    y = 90 + 0.05 * age + 0.002 * age**2 + 0.01 * temp
    d = pd.DataFrame({"tyre_age": age, "lap_time": y, "track_temp": temp})
    d.loc[3, "lap_time"] = np.nan
    r = tyre_degradation(d)
    assert abs(r["linear_deg"] - 0.05) < 0.03 and np.isfinite(r["temp_effect"])


def test_19_tyre_warmup():
    assert tyre_warmup_laps([92, 91, 90.3, 90.1, 90.0], 0.25) == 3


def test_20_fuel_correction():
    laps = np.arange(1, 6)
    observed = 90 - 0.035 * (laps - 1)
    corr = fuel_correct(observed, laps, -0.035)
    assert np.std(corr) < 1e-8


def test_21_track_evolution():
    d = pd.DataFrame(
        {
            "session_seconds": np.arange(100),
            "lap_time": np.r_[np.repeat(92.0, 50), np.repeat(90.0, 50)],
        }
    )
    c = track_evolution_correct(d, bins=2)
    assert abs(c.dropna().iloc[:40].median() - c.dropna().iloc[-40:].median()) < 0.1


def test_22_practice_pace():
    d = pd.DataFrame(
        {
            "driver": ["A"] * 6 + ["B"] * 6,
            "corrected_lap": [90, 90.1, 90.2, 91, 91.1, 91.2, 89, 89.1, 89.2, 90, 90.1, 90.2],
            "stint": [1] * 6 + [1] * 6,
        }
    )
    p = practice_pace(d)
    assert (
        p.loc[p.driver == "B", "single_lap_pace"].iloc[0]
        < p.loc[p.driver == "A", "single_lap_pace"].iloc[0]
    )


def test_23_dirty_air():
    d = pd.DataFrame({"pace_residual": [0, 0.1, 0.8, 0.9], "gap_ahead": [3, 4, 0.5, 1.5]})
    assert dirty_air_penalty(d) > 0.5 and traffic_class(0.5) == "dirty_lt1"


def test_24_overtake_model():
    rng = np.random.default_rng(0)
    n = 300
    pace = rng.normal(size=n)
    p = 1 / (1 + np.exp(-2 * pace))
    y = rng.random(n) < p
    d = pd.DataFrame(
        {
            "gap": rng.uniform(0.2, 2, n),
            "pace_delta": pace,
            "tyre_delta": rng.normal(size=n),
            "straight_length": rng.uniform(300, 1200, n),
            "drs_available": rng.integers(0, 2, n),
            "passed": y.astype(int),
        }
    )
    m, f = fit_overtake_model(d)
    assert m.coef_[0][f.index("pace_delta")] > 0


def test_25_reliability_prior_and_shift():
    d = pd.DataFrame({"car": ["A"] * 5, "dnf": [1, 0, 0, 0, 0]})
    r = reliability_features(d, window=3)
    assert r.iloc[0] != r.iloc[0] and 0 < r.iloc[-1] < 1


def test_26_strategy_features():
    assert pit_loss(100, 130, 8) == 22
    assert undercut_power(91, 90, 20, 2) == pytest.approx(0.1)


def test_27_backward_weather_and_resample():
    tel = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:01Z", "2026-01-01T00:03Z"]),
            "distance": [0, 10],
            "speed": [100, 120],
        }
    )
    w = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T00:02Z"]),
            "track_temp": [30, 31],
        }
    )
    out = backward_asof_weather(tel, w)
    assert list(out.track_temp) == [30, 31]
    rr = resample_telemetry(pd.DataFrame({"distance": [0, 10], "speed": [100, 120]}), 5)
    assert list(rr.speed) == [100, 110, 120]


def test_28_leakage_stage_guard():
    assert allowed("fp2", "POST_FP3") and not allowed("qualifying", "POST_FP3")
    with pytest.raises(ValueError):
        assert_no_future_features(["historical_form", "race_weather"], "POST_QUALI")


def test_29_timestamp_cutoff():
    good = pd.DataFrame({"f": ["2026-01-01T10:00Z"], "p": ["2026-01-01T11:00Z"]})
    assert assert_timestamp_cutoff(good, "f", "p")
    bad = pd.DataFrame({"f": ["2026-01-01T12:00Z"], "p": ["2026-01-01T11:00Z"]})
    with pytest.raises(ValueError):
        assert_timestamp_cutoff(bad, "f", "p")


def test_30_sources_and_simulator():
    u = openf1_url("car_data", session_key=123, driver_number=4)
    assert "session_key=123" in u and "driver_number=4" in u
    d = pd.DataFrame(
        {
            "driver": ["A", "B", "C"],
            "pace_score": [2, 1, 0],
            "dnf_prob": [0, 0, 0],
            "strategy_sd": [0, 0, 0],
        }
    )
    out = monte_carlo_race(d, 500, seed=1)
    assert out.loc[0, "win_probability"] == 1 and abs(out.win_probability.sum() - 1) < 1e-9
