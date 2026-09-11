"""Probability calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1pred.models.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    reliability_bins,
    winner_log_loss,
)


def _races(
    n_races: int = 200, n_drivers: int = 10, sharpness: float = 1.0, seed: int = 3
) -> pd.DataFrame:
    """Forecasts whose confidence is controlled, with outcomes drawn from truth."""
    rng = np.random.default_rng(seed)
    true_strength = np.linspace(1.5, -1.5, n_drivers)
    rows = []
    for r in range(n_races):
        true_p = np.exp(true_strength) / np.exp(true_strength).sum()
        winner = rng.choice(n_drivers, p=true_p)
        stated = np.power(true_p, sharpness)
        stated = stated / stated.sum()
        for d in range(n_drivers):
            rows.append(
                {
                    "race_id": f"r{r}",
                    "driver_id": f"d{d}",
                    "win_probability": stated[d],
                    "winner": int(d == winner),
                }
            )
    return pd.DataFrame(rows)


def test_temperature_one_is_the_identity():
    df = _races(20)
    out = apply_temperature(df["win_probability"], df["race_id"], 1.0)
    np.testing.assert_allclose(out.to_numpy(), df["win_probability"].to_numpy(), atol=1e-12)


def test_temperature_renormalises_each_race():
    df = _races(20)
    for t in (0.5, 2.0):
        out = apply_temperature(df["win_probability"], df["race_id"], t)
        sums = out.groupby(df["race_id"]).sum()
        np.testing.assert_allclose(sums.to_numpy(), 1.0, atol=1e-9)


def test_high_temperature_flattens_and_low_sharpens():
    df = _races(5)

    def spread(t):
        return apply_temperature(df["win_probability"], df["race_id"], t).max()

    assert spread(3.0) < spread(1.0) < spread(0.4)


def test_fit_recovers_a_flattening_temperature_for_overconfident_forecasts():
    """Stated probabilities sharper than truth must be corrected with T > 1."""
    df = _races(400, sharpness=2.0)
    fit = fit_temperature(df)
    assert fit.temperature > 1.05
    assert fit.improved


def test_fit_recovers_a_sharpening_temperature_for_underconfident_forecasts():
    df = _races(400, sharpness=0.5)
    fit = fit_temperature(df)
    assert fit.temperature < 0.95
    assert fit.improved


def test_fit_leaves_a_well_calibrated_forecast_alone():
    df = _races(400, sharpness=1.0)
    fit = fit_temperature(df)
    assert fit.temperature == pytest.approx(1.0, abs=0.35)


def test_winner_log_loss_rewards_confident_correct_forecasts():
    good = pd.DataFrame({"race_id": ["r"] * 2, "p": [0.9, 0.1], "y": [True, False]})
    bad = pd.DataFrame({"race_id": ["r"] * 2, "p": [0.1, 0.9], "y": [True, False]})
    assert winner_log_loss(good["p"], good["race_id"], good["y"]) < winner_log_loss(
        bad["p"], bad["race_id"], bad["y"]
    )


def test_log_loss_skips_races_without_exactly_one_winner():
    df = pd.DataFrame(
        {
            "race_id": ["a", "a", "b", "b"],
            "p": [0.5, 0.5, 0.5, 0.5],
            "y": [True, False, False, False],
        }
    )
    assert np.isfinite(winner_log_loss(df["p"], df["race_id"], df["y"]))


def test_reliability_bins_detect_overconfidence():
    df = _races(400, sharpness=2.5)
    table = reliability_bins(df, prob_col="win_probability", outcome_col="winner")
    top = table.iloc[-1]
    assert top["mean_predicted"] > top["observed_frequency"]
    assert expected_calibration_error(df, prob_col="win_probability", outcome_col="winner") > 0.0


def test_ece_is_near_zero_for_a_calibrated_forecast():
    df = _races(1500, sharpness=1.0)
    assert expected_calibration_error(df, prob_col="win_probability", outcome_col="winner") < 0.05
