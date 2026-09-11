"""Monte Carlo simulator: probability coherence, determinism, and circuit effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pred.sim.race import (
    SimulationSettings,
    overtaking_blend,
    position_to_latent,
    simulate_race,
)


def _entries(n: int = 20, spread: float = 1.5, dnf: float = 0.10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver_id": [f"d{i:02d}" for i in range(n)],
            "constructor_id": [f"t{i // 2}" for i in range(n)],
            "strength": np.linspace(spread, -spread, n),
            "dnf_probability": dnf,
        }
    )


@pytest.fixture
def forecast():
    return simulate_race(
        _entries(), overtaking_difficulty=0.5, settings=SimulationSettings(n_sims=4000)
    )


def test_win_probabilities_sum_to_one(forecast):
    assert forecast.table["win_probability"].sum() == pytest.approx(1.0, abs=1e-9)


def test_podium_and_top10_masses_are_exact(forecast):
    assert forecast.table["podium_probability"].sum() == pytest.approx(3.0, abs=1e-9)
    assert forecast.table["top10_probability"].sum() == pytest.approx(10.0, abs=1e-9)


def test_all_probabilities_are_in_the_unit_interval(forecast):
    for col in (
        "win_probability",
        "podium_probability",
        "top10_probability",
        "dnf_probability_sim",
    ):
        values = forecast.table[col]
        assert (values >= 0).all() and (values <= 1).all(), col


def test_expected_finish_positions_average_to_the_field_centre(forecast):
    n = len(forecast.table)
    assert forecast.table["expected_finish"].mean() == pytest.approx((n + 1) / 2, abs=1e-9)


def test_stronger_entries_win_more_often(forecast):
    t = forecast.table.sort_values("strength", ascending=False)
    win = t["win_probability"].to_numpy()
    assert win[0] > win[len(win) // 2] >= win[-1]
    assert (
        t["expected_finish"].iloc[0]
        < t["expected_finish"].iloc[len(win) // 2]
        < t["expected_finish"].iloc[-1]
    )


def test_simulated_dnf_rate_matches_the_input_hazard():
    """Shared attrition must preserve the advertised marginal probability."""
    base = 0.10
    f = simulate_race(
        _entries(dnf=base), overtaking_difficulty=0.5, settings=SimulationSettings(n_sims=20000)
    )
    realised = f.table["dnf_probability_sim"].mean()
    expected = base
    assert realised == pytest.approx(expected, abs=0.01)


def test_results_are_reproducible_under_the_same_seed():
    a = simulate_race(
        _entries(), overtaking_difficulty=0.4, settings=SimulationSettings(n_sims=3000)
    )
    b = simulate_race(
        _entries(), overtaking_difficulty=0.4, settings=SimulationSettings(n_sims=3000)
    )
    pd.testing.assert_series_equal(a.table["win_probability"], b.table["win_probability"])


def test_different_seeds_give_different_draws():
    a = simulate_race(
        _entries(), overtaking_difficulty=0.4, settings=SimulationSettings(n_sims=3000, seed=1)
    )
    b = simulate_race(
        _entries(), overtaking_difficulty=0.4, settings=SimulationSettings(n_sims=3000, seed=2)
    )
    assert not np.allclose(a.table["win_probability"], b.table["win_probability"])


def test_overtaking_blend_is_monotonic_and_bounded():
    values = [overtaking_blend(d) for d in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert values == sorted(values)
    assert 0.0 < values[0] and values[-1] < 1.0


def test_hard_circuit_preserves_grid_order_more_than_easy_one():
    """The defining property: P5 at Monaco is not P5 at Monza."""
    entries = _entries(n=20, dnf=0.0)
    entries["grid"] = np.arange(1, 21)
    # Give everyone identical pace so only the grid weight can matter.
    entries["strength"] = 0.0
    easy = simulate_race(
        entries,
        overtaking_difficulty=0.05,
        settings=SimulationSettings(n_sims=8000),
        grid_col="grid",
    )
    hard = simulate_race(
        entries,
        overtaking_difficulty=0.95,
        settings=SimulationSettings(n_sims=8000),
        grid_col="grid",
    )
    pole_easy = easy.table.loc[easy.table["grid"] == 1, "win_probability"].iloc[0]
    pole_hard = hard.table.loc[hard.table["grid"] == 1, "win_probability"].iloc[0]
    assert pole_hard > pole_easy


def test_known_grid_suppresses_pole_probability_column():
    entries = _entries()
    entries["grid"] = np.arange(1, len(entries) + 1)
    known = simulate_race(
        entries,
        overtaking_difficulty=0.5,
        settings=SimulationSettings(n_sims=2000),
        grid_col="grid",
    )
    unknown = simulate_race(
        entries, overtaking_difficulty=0.5, settings=SimulationSettings(n_sims=2000)
    )
    assert known.grid_known and "pole_probability" not in known.table
    assert not unknown.grid_known and "pole_probability" in unknown.table
    assert unknown.table["pole_probability"].sum() == pytest.approx(1.0, abs=1e-9)


def test_rookie_uncertainty_widens_the_finishing_distribution():
    entries = _entries(n=10)
    entries["strength"] = 0.0
    entries["strength_sd"] = [0.0] * 9 + [2.0]
    f = simulate_race(entries, overtaking_difficulty=0.5, settings=SimulationSettings(n_sims=10000))
    t = f.table.set_index("driver_id")
    uncertain = t.loc["d09"]
    typical = t.loc["d00"]
    assert (uncertain["finish_p90"] - uncertain["finish_p10"]) >= (
        typical["finish_p90"] - typical["finish_p10"]
    )


def test_position_to_latent_matches_the_results_scale():
    """Grid slots and race results must share units or the blend is meaningless."""
    from f1pred.features.pace import normal_scores

    grid = position_to_latent(np.array([1, 10, 20]), 20)
    race = normal_scores(pd.Series([1, 10, 20]), pd.Series([20, 20, 20])).to_numpy()
    np.testing.assert_allclose(grid, race, atol=1e-12)


def test_expected_finish_if_classified_is_tie_free_and_better_than_unconditional():
    f = simulate_race(
        _entries(), overtaking_difficulty=0.5, settings=SimulationSettings(n_sims=6000)
    )
    col = f.table["expected_finish_if_classified"]
    assert col.nunique() == len(col), "ranking key must not tie"
    assert (col <= f.table["expected_finish"] + 1e-9).all()


def test_empty_entry_list_is_rejected():
    with pytest.raises(ValueError, match="no entries"):
        simulate_race(_entries(0), overtaking_difficulty=0.5)


def test_variance_budget_keeps_the_constants_the_evidence_supports():
    """A fitted budget was measured and rejected; the defaults must not drift to it.

    Refitting the four spread terms against a measured strength posterior looked
    like a 0.076 gain until the control was corrected: against the real incumbent
    it cost 0.219 nats of winner log-loss. The numbers stay where the evidence
    put them, and the fitted values stay in config as a recorded negative result.
    """
    from f1pred.config import FITTED_VARIANCE_BUDGET

    settings = SimulationSettings()
    assert settings.qualifying_noise == pytest.approx(0.55)
    assert settings.race_pace_noise == pytest.approx(0.60)
    assert settings.start_swing_slots == pytest.approx(1.6)
    assert settings.strategy_swing_slots == pytest.approx(2.0)
    for name, rejected in FITTED_VARIANCE_BUDGET.items():
        assert getattr(settings, name) != pytest.approx(rejected, abs=1e-4), name


def test_estimation_uncertainty_widens_the_forecast():
    """Strength uncertainty must widen a distribution, never shift its centre."""
    entries = pd.DataFrame(
        {
            "driver_id": ["a", "b", "c", "d"],
            "constructor_id": ["t1", "t1", "t2", "t2"],
            "strength": [1.0, 0.5, -0.5, -1.0],
            "dnf_probability": [0.1] * 4,
        }
    )
    settings = SimulationSettings(n_sims=20000, seed=11)
    confident = simulate_race(
        entries.assign(strength_sd=0.05), overtaking_difficulty=0.5, settings=settings
    ).table.set_index("driver_id")
    unsure = simulate_race(
        entries.assign(strength_sd=0.9), overtaking_difficulty=0.5, settings=settings
    ).table.set_index("driver_id")
    # The favourite stays the favourite, but is less of one.
    assert confident.loc["a", "win_probability"] > unsure.loc["a", "win_probability"]
    assert unsure.loc["d", "win_probability"] > confident.loc["d", "win_probability"]
    assert unsure["win_probability"].idxmax() == "a"
