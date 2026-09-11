"""DNF hazard model: shrinkage behaviour, hierarchy, and leakage."""

from __future__ import annotations

import pandas as pd
import pytest

from f1pred.features.reliability import (
    STREET_CIRCUITS,
    beta_rate,
    circuit_dnf_multipliers,
    fit_reliability,
    track_type_multiplier,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["started"] = 1
    df["regime"] = "NEW_PU_AERO_2026"
    return df


def test_beta_rate_shrinks_toward_the_prior():
    """Zero failures in ten starts must not imply zero risk."""
    assert 0.0 < beta_rate(0, 10, 1.2, 12.0) < 0.10
    assert beta_rate(0, 10, 1.2, 12.0) < beta_rate(3, 10, 1.2, 12.0)


def test_beta_rate_converges_to_the_empirical_rate_with_data():
    assert beta_rate(200, 1000, 1.2, 12.0) == pytest.approx(0.2, abs=0.01)


def test_reliable_and_unreliable_teams_separate():
    rows = []
    for i in range(30):
        rows.append(
            {
                "driver_id": f"good{i % 2}",
                "constructor_id": "solid",
                "dnf": 0,
                "event_index": float(i),
            }
        )
        rows.append(
            {
                "driver_id": f"bad{i % 2}",
                "constructor_id": "fragile",
                "dnf": int(i % 2 == 0),
                "event_index": float(i),
            }
        )
    est = fit_reliability(_frame(rows), as_of_index=100.0, target_regime="NEW_PU_AERO_2026")
    rates = est.teams.set_index("constructor_id")["team_dnf_rate"]
    assert rates["fragile"] > rates["solid"]
    assert est.hazard("bad0", "fragile") > est.hazard("good0", "solid")


def test_driver_excess_is_identified_against_the_team_mate():
    """Same car, one driver crashing: the excess must attach to the driver."""
    rows = []
    for i in range(40):
        rows.append(
            {"driver_id": "careful", "constructor_id": "t", "dnf": 0, "event_index": float(i)}
        )
        rows.append(
            {"driver_id": "crasher", "constructor_id": "t", "dnf": 1, "event_index": float(i)}
        )
    est = fit_reliability(_frame(rows), as_of_index=100.0, target_regime="NEW_PU_AERO_2026")
    excess = est.drivers.set_index("driver_id")["driver_excess"]
    assert excess["crasher"] > 0 > excess["careful"]
    assert est.hazard("crasher", "t") > est.hazard("careful", "t")


def test_hazard_stays_in_bounds_under_extreme_multipliers():
    rows = [
        {"driver_id": "d", "constructor_id": "t", "dnf": 1, "event_index": float(i)}
        for i in range(20)
    ]
    est = fit_reliability(_frame(rows), as_of_index=100.0, target_regime="NEW_PU_AERO_2026")
    assert 0.0 < est.hazard("d", "t", multiplier=50.0) <= 0.60
    assert 0.0 < est.hazard("d", "t", multiplier=0.001) >= 0.01


def test_unknown_entry_falls_back_to_the_base_rate():
    rows = [
        {"driver_id": "d", "constructor_id": "t", "dnf": i % 4 == 0, "event_index": float(i)}
        for i in range(20)
    ]
    est = fit_reliability(_frame(rows), as_of_index=100.0, target_regime="NEW_PU_AERO_2026")
    fallback = est.hazard("never_seen", "never_seen_either")
    assert 0.01 <= fallback <= 0.60


def test_fit_ignores_races_at_or_after_the_cutoff():
    early = [
        {"driver_id": "d", "constructor_id": "t", "dnf": 0, "event_index": float(i)}
        for i in range(10)
    ]
    late = [
        {"driver_id": "d", "constructor_id": "t", "dnf": 1, "event_index": float(i)}
        for i in range(10, 30)
    ]
    clean = fit_reliability(_frame(early), as_of_index=10.0, target_regime="NEW_PU_AERO_2026")
    withfuture = fit_reliability(
        _frame(early + late), as_of_index=10.0, target_regime="NEW_PU_AERO_2026"
    )
    assert clean.hazard("d", "t") == pytest.approx(withfuture.hazard("d", "t"))


def test_street_circuits_are_more_attritional_than_permanent_ones():
    rows = []
    for i in range(200):
        street = i % 2 == 0
        rows.append(
            {
                "driver_id": f"d{i % 5}",
                "constructor_id": f"t{i % 3}",
                "circuit_id": "monaco" if street else "monza",
                "dnf": int(street and i % 4 == 0),
                "event_index": float(i),
            }
        )
    df = _frame(rows)
    assert track_type_multiplier(df, is_street=True) > track_type_multiplier(df, is_street=False)


def test_circuit_multipliers_shrink_toward_one_with_little_history():
    rows = []
    for i in range(400):
        rows.append(
            {
                "driver_id": f"d{i % 10}",
                "constructor_id": f"t{i % 5}",
                "circuit_id": "common",
                "dnf": int(i % 10 == 0),
                "event_index": float(i),
            }
        )
    # A circuit visited twice with total carnage must not earn an extreme multiplier.
    for _i in range(2):
        rows.append(
            {
                "driver_id": "d0",
                "constructor_id": "t0",
                "circuit_id": "rare",
                "dnf": 1,
                "event_index": 500.0,
            }
        )
    out = circuit_dnf_multipliers(_frame(rows)).set_index("circuit_id")
    assert out.loc["rare", "dnf_multiplier"] < 3.0


def test_monaco_and_madrid_are_registered_as_street_circuits():
    assert "monaco" in STREET_CIRCUITS
    assert "madring" in STREET_CIRCUITS
