"""Statistical contracts for the experimental hierarchy, independent of race scores."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import helmert

from f1pred.models.retirement import RetirementPosterior, fit_retirement
from f1pred.models.state_space import (
    Observation,
    PaceFilter,
    VarianceComponents,
    contrast_covariance,
    filtered_state,
    fit_variances,
    gaussian_update,
    sample_ranking,
    timing_observations,
)

V = VarianceComponents(1.0, 0.4, 0.05, 0.002, 0.2, 0.3)


def entries():
    return pd.DataFrame(
        {"driver_id": ["a", "b", "c", "d"], "constructor_id": ["red", "red", "blue", "blue"]}
    )


def observations(n=18, seed=91):
    rng = np.random.default_rng(seed)
    roster = entries()
    state = PaceFilter(["red", "blue"], list("abcd"), V)
    h = state.design(roster)
    x = rng.multivariate_normal(state.mean, state.covariance)
    rows = []
    for idx in range(n):
        x += rng.normal(size=len(x)) * np.sqrt([V.team_step] * 2 + [V.driver_step] * 4)
        for kind in ("qualifying", "race"):
            y = h @ x + rng.normal(size=4) * np.sqrt(getattr(V, kind)) + 200
            rows.append(
                Observation(f"2020_{idx + 1:02}", float(idx), 2020, kind, roster, y, roster)
            )
    return rows


def test_kalman_update_equals_independent_batch_posterior():
    rng = np.random.default_rng(50)
    x = rng.normal(size=(10, 4))
    y = rng.normal(size=10)
    prior = np.diag([1, 2, 3, 4])
    mean, cov, _ = gaussian_update(np.zeros(4), prior, x, y, 0.4)
    expected_cov = np.linalg.inv(np.linalg.inv(prior) + x.T @ x / 0.4)
    np.testing.assert_allclose(cov, expected_cov, atol=1e-12)
    np.testing.assert_allclose(mean, expected_cov @ x.T @ y / 0.4, atol=1e-12)


def test_identical_teammates_do_not_get_spurious_opposite_effects():
    roster = entries()
    state = PaceFilter(["red", "blue"], list("abcd"), V)
    for idx in range(20):
        state.observe(
            Observation(
                str(idx), idx, 2020, "race", roster, np.array([2.0, 2.0, -2.0, -2.0]), roster
            )
        )
    np.testing.assert_allclose(state.mean[list(state.drivers.values())], 0, atol=1e-12)
    assert state.mean[state.teams["red"]] > state.mean[state.teams["blue"]]


def test_driver_columns_are_centered_within_the_full_team_roster():
    roster = entries()
    state = PaceFilter(["red", "blue"], list("abcd"), V)
    h = state.design(roster)
    np.testing.assert_allclose(h[:2, list(state.drivers.values())].sum(axis=0), 0)
    # One teammate missing a time must not erase the contrast that was measured.
    np.testing.assert_allclose(state.design(roster.iloc[:1], roster), h[:1])


def test_common_uncertainty_cancels_but_contrast_uncertainty_remains():
    covariance = np.array([[4.0, 3.0], [3.0, 5.0]])
    expected_pair_variance = 4 + 5 - 2 * 3
    contrast = contrast_covariance(covariance)
    np.testing.assert_allclose(contrast, contrast_covariance(covariance + 1000), atol=1e-12)
    assert contrast[0, 0] + contrast[1, 1] - 2 * contrast[0, 1] == pytest.approx(
        expected_pair_variance
    )
    assert np.diag(covariance).sum() != pytest.approx(expected_pair_variance)
    rng = np.random.default_rng(12)
    draws = rng.multivariate_normal([0, 0], contrast, size=100000)
    assert np.var(draws[:, 0] - draws[:, 1]) == pytest.approx(expected_pair_variance, rel=0.015)


def test_session_offset_does_not_change_filtered_state():
    a, b = (PaceFilter(["red", "blue"], list("abcd"), V) for _ in range(2))
    for o in observations():
        a.observe(o)
        b.observe(replace(o, values=o.values + 700))
    np.testing.assert_allclose(a.mean, b.mean, atol=1e-11)
    np.testing.assert_allclose(a.covariance, b.covariance, atol=1e-12)


def test_relabeling_and_row_order_do_not_change_inference():
    a, b = (PaceFilter(["red", "blue"], list("abcd"), V) for _ in range(2))
    order = np.array([3, 1, 0, 2])
    for o in observations():
        a.observe(o)
        b.observe(replace(o, entries=o.entries.iloc[order], values=o.values[order]))
    np.testing.assert_allclose(a.mean, b.mean, atol=1e-10)
    np.testing.assert_allclose(a.covariance, b.covariance, atol=1e-10)


def test_dynamics_widen_uncertainty_and_adapt_to_team_changes():
    roster = entries()
    state = PaceFilter(["red", "blue"], list("abcd"), V)
    for idx in range(20):
        y = np.array([1, 1, -1, -1]) * (1 if idx < 10 else -1)
        state.observe(Observation(str(idx), idx, 2020, "race", roster, y, roster))
    assert state.mean[state.teams["red"]] < state.mean[state.teams["blue"]]
    _, before, _ = state.predict(roster, event_index=20, season=2020)
    _, after, _ = state.predict(roster, event_index=25, season=2020)
    assert np.trace(after) > np.trace(before)
    with pytest.raises(ValueError, match="backwards"):
        state.advance(19, 2020)


def test_regulation_reset_preserves_drivers_and_resets_teams():
    state = PaceFilter(["red", "blue"], list("abcd"), V)
    for o in observations():
        state.observe(o)
    drivers = state.mean[list(state.drivers.values())].copy()
    state.advance(30, 2022)
    np.testing.assert_allclose(state.mean[list(state.teams.values())], 0)
    np.testing.assert_allclose(state.mean[list(state.drivers.values())], drivers)
    np.testing.assert_allclose(np.diag(state.covariance)[list(state.teams.values())], V.team_prior)


def test_new_driver_has_a_prior_not_an_arbitrary_penalty():
    state = PaceFilter(["red", "blue"], list("abcd"), V)
    for o in observations():
        state.observe(o)
    roster = entries().replace({"d": "rookie"})
    state.predict(roster, event_index=20, season=2020)
    i = state.drivers["rookie"]
    assert state.mean[i] == 0
    assert state.covariance[i, i] == V.driver_prior


def test_synthetic_posterior_contrast_intervals_have_nominal_coverage():
    # Repeated data generated from the declared prior: validates the inference,
    # not the real-world Gaussian assumption or empirical-Bayes plug-in widths.
    rng = np.random.default_rng(9)
    state = PaceFilter(["red", "blue"], list("abcd"), V)
    c = helmert(4, full=False)
    h = c @ state.design(entries())
    contrast = np.array([1.0, -1.0, 0.0, 0.0]) @ state.design(entries())
    covered = 0
    for _ in range(600):
        truth = rng.multivariate_normal(state.mean, state.covariance)
        y = h @ truth + rng.normal(size=3) * np.sqrt(V.race)
        mean, cov, _ = gaussian_update(state.mean, state.covariance, h, y, V.race)
        covered += abs(contrast @ (truth - mean)) <= 1.644853627 * np.sqrt(
            contrast @ cov @ contrast
        )
    assert 0.86 < covered / 600 < 0.94


def test_variance_fit_and_state_ignore_future_observations():
    prior = observations()
    fit = fit_variances(prior, as_of_index=18)
    corrupted = [
        *prior,
        *[replace(o, event_index=o.event_index + 100, values=o.values * 1000) for o in prior],
    ]
    again = fit_variances(corrupted, as_of_index=18)
    assert fit == again
    assert fit.last_training_index < fit.as_of_index
    a, b = [filtered_state(obs, fit, as_of_index=18) for obs in (prior, corrupted)]
    np.testing.assert_allclose(a.mean, b.mean)
    with pytest.raises(ValueError, match="future"):
        filtered_state(prior, fit, as_of_index=17)


def test_timing_measurements_do_not_mix_segments_or_fastest_laps():
    roster = entries()
    result = roster.assign(
        race_id="r",
        season=2020,
        event_index=1,
        classified=1,
        laps=[50, 50, 49, 50],
        total_ms=[5000000, 5050000, np.nan, 5100000],
        fastest_lap_s=1,
    )
    qual = roster.assign(race_id="r", q1_s=[90, 91, 92, 93], q2_s=1, q3_s=1)
    obs = timing_observations(result, qual)
    race = next(o for o in obs if o.kind == "race")
    q1 = next(o for o in obs if o.kind == "qualifying")
    assert len(race.entries) == 3 and "c" not in set(race.entries.driver_id)
    assert len(race.roster) == 4
    np.testing.assert_allclose(q1.values, -100 * np.log([90, 91, 92, 93]))
    np.testing.assert_allclose(race.values, -100 * np.log([100, 101, 102]))


def test_sampler_is_reproducible_and_returns_coherent_rank_probabilities():
    retirement = RetirementPosterior(2, 18, {"red": (3, 27)}, -1, True, False)
    kw = dict(n_sims=20000, retirement=retirement)
    a = sample_ranking(entries(), np.array([1.0, 0.5, 0, -0.5]), np.eye(4), **kw)
    b = sample_ranking(entries(), np.array([1.0, 0.5, 0, -0.5]), np.eye(4), **kw)
    pd.testing.assert_frame_equal(a, b)
    assert a.win_probability.sum() == pytest.approx(1)
    assert a.podium_probability.sum() <= 3
    assert a.top10_probability.sum() == pytest.approx(4 - a.dnf_probability_sim.sum())
    assert (abs(a.dnf_probability_sim - a.dnf_probability) < 0.01).all()
    assert (a.finish_p10 <= a.finish_p50).all() and (a.finish_p50 <= a.finish_p90).all()
    with pytest.raises(ValueError, match="positive semidefinite"):
        sample_ranking(entries(), np.zeros(4), -np.eye(4))


def test_retirement_prior_is_training_only_and_regime_local():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "event_index": np.repeat(np.arange(40), 8),
            "constructor_id": np.tile(np.repeat(list("abcd"), 2), 40),
            "regime": "old",
            "started": 1,
            "dnf": rng.binomial(1, 0.15, size=320),
        }
    )
    fit = fit_retirement(frame, as_of_index=30, target_regime="new")
    assert not fit.teams
    frame.loc[frame.event_index >= 30, "dnf"] = 1
    assert fit == fit_retirement(frame, as_of_index=30, target_regime="new")
    assert fit.last_training_index == 29


def test_retirements_cannot_count_as_podiums_or_points():
    posterior = RetirementPosterior(5, 5, {}, -1, True, False)
    result = sample_ranking(entries(), np.zeros(4), np.eye(4), n_sims=40000, retirement=posterior)
    assert result.win_probability.sum() == pytest.approx(1)
    assert (result.win_probability <= result.podium_probability).all()
    assert (result.podium_probability <= result.top10_probability).all()
    np.testing.assert_allclose(result.top10_probability, 1 - result.dnf_probability_sim)
    assert (abs(result.dnf_probability - result.dnf_probability_sim) < 0.01).all()


def test_known_future_names_do_not_age_priors_before_debut():
    # Batch fitting knows all earlier names; a live filter discovers them as
    # they enter. Both must give identical states, including rookie uncertainty.
    fixed = PaceFilter(["red", "blue"], list("abcde"), V)
    growing = PaceFilter([], [], V)
    for observation in observations(12):
        if observation.event_index >= 6:
            observation = replace(
                observation,
                entries=observation.entries.replace({"d": "e"}),
                roster=observation.roster.replace({"d": "e"}),
            )
        fixed.observe(observation)
        growing.observe(observation)
    names = sorted(growing.teams) + sorted(growing.drivers)
    left = [{**fixed.teams, **fixed.drivers}[name] for name in names]
    right = [{**growing.teams, **growing.drivers}[name] for name in names]
    np.testing.assert_allclose(fixed.mean[left], growing.mean[right], atol=1e-11)
    np.testing.assert_allclose(
        fixed.covariance[np.ix_(left, left)], growing.covariance[np.ix_(right, right)], atol=1e-11
    )


def test_preweekend_prediction_refuses_an_already_observed_race():
    state = PaceFilter([], [], V)
    observation = observations(1)[0]
    state.observe(observation)
    with pytest.raises(ValueError, match="this race's observations"):
        state.predict(entries(), event_index=observation.event_index, season=2020)
