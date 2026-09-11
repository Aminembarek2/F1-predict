"""Experimental hierarchical pace model with chronological Gaussian filtering.

Observations are within-session contrasts of log times (percentage units).
Team effects describe team-and-lineup pace; driver effects describe differences
from the current teammate mean. They are not causal car/driver ratings.
Variance components are estimated by training-only marginal likelihood. The
posterior is conditional on those empirical-Bayes estimates, not full Bayes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, helmert
from scipy.optimize import minimize

from ..config import SEED, regime_for
from .retirement import RetirementPosterior


@dataclass(frozen=True)
class VarianceComponents:
    team_prior: float
    driver_prior: float
    team_step: float
    driver_step: float
    qualifying: float
    race: float

    def __post_init__(self) -> None:
        if not all(np.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError("variance components must be finite and positive")


@dataclass(frozen=True)
class Observation:
    race_id: str
    event_index: float
    season: int
    kind: str
    entries: pd.DataFrame
    values: np.ndarray
    # Full weekend roster, before selecting usable timing measurements.
    roster: pd.DataFrame


def timing_observations(results: pd.DataFrame, qualifying: pd.DataFrame) -> list[Observation]:
    """Use Q1 and lead-lap elapsed race time, with no fastest-lap/rank fallback.

    Q1 avoids comparing drivers in different qualifying segments. Race elapsed
    time includes traffic, stops and neutralisations; it is an outcome proxy,
    not clean-air or fuel-corrected pace. Missing/lapped cars are excluded and
    coverage must be reported. The selection is not missing at random.
    """
    observations = []
    for race_id, result in results.groupby("race_id", sort=False):
        roster = (
            pd.concat(
                [
                    result[["driver_id", "constructor_id"]],
                    qualifying.loc[
                        qualifying["race_id"].eq(race_id), ["driver_id", "constructor_id"]
                    ],
                ]
            )
            .drop_duplicates()
            .sort_values("driver_id")
            .reset_index(drop=True)
        )
        if roster["driver_id"].duplicated().any():
            raise ValueError("conflicting constructors for a driver at one event")
        for kind, frame in (
            ("qualifying", qualifying.loc[qualifying["race_id"].eq(race_id)]),
            ("race", result),
        ):
            frame = frame.copy().sort_values("driver_id")
            if frame.duplicated("driver_id").any():
                raise ValueError("duplicate driver timing")
            if kind == "qualifying":
                times = pd.to_numeric(frame["q1_s"], errors="coerce")
            else:
                laps = pd.to_numeric(frame["laps"], errors="coerce")
                total = pd.to_numeric(frame["total_ms"], errors="coerce")
                times = (total / (1000 * laps.where(laps > 0))).where(
                    frame["classified"].eq(1) & laps.eq(laps.max())
                )
            good = np.isfinite(times) & times.gt(0)
            if good.sum() < 3:
                continue
            observations.append(
                Observation(
                    str(race_id),
                    float(result["event_index"].iloc[0]),
                    int(result["season"].iloc[0]),
                    kind,
                    frame.loc[good, ["driver_id", "constructor_id"]].reset_index(drop=True),
                    -100 * np.log(times.loc[good].to_numpy(float)),
                    roster,
                )
            )
    return sorted(observations, key=lambda o: (o.event_index, o.kind != "qualifying"))


def contrast_covariance(covariance: np.ndarray) -> np.ndarray:
    """Covariance after removing the common field offset; retain correlations."""
    c = np.asarray(covariance, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.isfinite(c).all():
        raise ValueError("covariance must be a finite square matrix")
    if not np.allclose(c, c.T, atol=1e-9):
        raise ValueError("covariance must be symmetric")
    return c - c.mean(axis=0)[None, :] - c.mean(axis=1)[:, None] + c.mean()


def gaussian_update(mean, covariance, design, values, noise):
    """Kalman update and innovation likelihood, using Cholesky solves."""
    cross = covariance @ design.T
    innovation = values - design @ mean
    s = design @ cross + np.eye(len(values)) * noise
    factor = cho_factor(s, lower=True, check_finite=False)
    solved = cho_solve(factor, innovation, check_finite=False)
    updated_mean = mean + cross @ solved
    updated_cov = covariance - cross @ cho_solve(factor, cross.T, check_finite=False)
    updated_cov = (updated_cov + updated_cov.T) / 2
    nll = 0.5 * (
        len(values) * np.log(2 * np.pi) + 2 * np.log(np.diag(factor[0])).sum() + innovation @ solved
    )
    return updated_mean, updated_cov, float(nll)


class PaceFilter:
    """Random-walk team and driver states with proper group-level priors.

    Constructors restart at regulation boundaries. Drivers persist. New entities
    receive the fitted group prior; no rookie penalty or observation-count
    shrinkage is added. All states share fitted group variance components.
    """

    def __init__(self, teams: list[str], drivers: list[str], variances: VarianceComponents):
        self.teams = {name: i for i, name in enumerate(sorted(set(teams)))}
        self.drivers = {name: len(self.teams) + i for i, name in enumerate(sorted(set(drivers)))}
        self.variances = variances
        self.mean = np.zeros(len(self.teams) + len(self.drivers))
        self.covariance = np.diag(
            [variances.team_prior] * len(self.teams) + [variances.driver_prior] * len(self.drivers)
        )
        self.event_index: float | None = None
        self.regime: str | None = None
        self.last_observed_index: float | None = None
        self.active_teams: set[str] = set()
        self.active_drivers: set[str] = set()

    def _ensure_entries(self, entries: pd.DataFrame) -> None:
        # Append new state dimensions without changing existing indices.
        for mapping, col, prior in (
            (self.teams, "constructor_id", self.variances.team_prior),
            (self.drivers, "driver_id", self.variances.driver_prior),
        ):
            for name in sorted(set(entries[col]) - set(mapping)):
                mapping[name] = len(self.mean)
                self.mean = np.append(self.mean, 0.0)
                self.covariance = np.pad(self.covariance, ((0, 1), (0, 1)))
                self.covariance[-1, -1] = prior
        self.active_teams.update(entries["constructor_id"])
        self.active_drivers.update(entries["driver_id"])

    def advance(self, event_index: float, season: int) -> None:
        if not np.isfinite(event_index):
            raise ValueError("event index must be finite")
        if self.event_index is not None and event_index < self.event_index:
            raise ValueError("filter cannot move backwards in time")
        delta = 0 if self.event_index is None else event_index - self.event_index
        t = [self.teams[name] for name in self.active_teams]
        d = [self.drivers[name] for name in self.active_drivers]
        self.covariance[t, t] += delta * self.variances.team_step
        self.covariance[d, d] += delta * self.variances.driver_step
        regime = regime_for(season)
        if self.regime is not None and regime != self.regime:
            self.mean[t] = 0
            self.covariance[t, :] = 0
            self.covariance[:, t] = 0
            self.covariance[t, t] = self.variances.team_prior
        self.event_index, self.regime = event_index, regime

    def design(self, entries: pd.DataFrame, roster: pd.DataFrame | None = None) -> np.ndarray:
        roster = entries if roster is None else roster
        if (
            entries[["driver_id", "constructor_id"]].isna().any().any()
            or entries["driver_id"].duplicated().any()
        ):
            raise ValueError("unique drivers and nonmissing constructors required")
        if (
            roster["driver_id"].duplicated().any()
            or roster[["driver_id", "constructor_id"]].isna().any().any()
        ):
            raise ValueError("invalid roster")
        membership = set(map(tuple, roster[["driver_id", "constructor_id"]].to_numpy()))
        if not all(
            tuple(row) in membership for row in entries[["driver_id", "constructor_id"]].to_numpy()
        ):
            raise ValueError("timed entries must belong to the full roster")
        self._ensure_entries(roster)
        h = np.zeros((len(entries), len(self.mean)))
        lineups = roster.groupby("constructor_id")["driver_id"].agg(list).to_dict()
        for i, row in enumerate(entries.itertuples(index=False)):
            h[i, self.teams[row.constructor_id]] = 1
            h[i, self.drivers[row.driver_id]] = 1
            mates = lineups[row.constructor_id]
            for mate in mates:
                h[i, self.drivers[mate]] -= 1 / len(mates)
        return h

    def observe(self, observation: Observation) -> float:
        self.advance(observation.event_index, observation.season)
        h = self.design(observation.entries, observation.roster)
        c = helmert(len(h), full=False)
        self.mean, self.covariance, nll = gaussian_update(
            self.mean,
            self.covariance,
            c @ h,
            c @ observation.values,
            getattr(self.variances, observation.kind),
        )
        self.last_observed_index = observation.event_index
        return nll

    def predict(self, entries: pd.DataFrame, *, event_index: float, season: int):
        """Mean, latent contrast covariance, and race predictive covariance."""
        if self.last_observed_index is not None and self.last_observed_index >= event_index:
            raise ValueError("pre-weekend prediction cannot use this race's observations")
        self.advance(event_index, season)
        h = self.design(entries)
        mean = h @ self.mean
        latent = contrast_covariance(h @ self.covariance @ h.T)
        predictive = contrast_covariance(latent + np.eye(len(h)) * self.variances.race)
        return mean - mean.mean(), latent, predictive


@dataclass(frozen=True)
class VarianceFit:
    variances: VarianceComponents
    as_of_index: float
    last_training_index: float
    n_events: int
    negative_log_likelihood: float
    converged: bool
    starts: tuple[dict, ...]
    at_boundary: tuple[str, ...]


def fit_variances(
    observations: list[Observation], *, as_of_index: float, maxiter: int = 150
) -> VarianceFit:
    """Estimate six shared variance components on earlier observations only.

    Two fixed starting points test numerical sensitivity, not forecast selection.
    Bounds (variance 1e-6 to 1e3 percentage-points squared) are numerical; active
    bounds and optimizer convergence are reported for every training fold.
    """
    prior = [o for o in observations if o.event_index < as_of_index]
    if len({o.race_id for o in prior}) < 10 or {o.kind for o in prior} != {"qualifying", "race"}:
        raise ValueError("at least ten earlier events and both timing sources required")
    roster = pd.concat([o.roster for o in prior]).drop_duplicates()
    teams, drivers = sorted(roster.constructor_id.unique()), sorted(roster.driver_id.unique())
    # Precompute immutable measurement matrices for repeated likelihood calls.
    template = PaceFilter(teams, drivers, VarianceComponents(1, 1, 1, 1, 1, 1))
    measurements = []
    for o in prior:
        c = helmert(len(o.entries), full=False)
        measurements.append((o, c @ template.design(o.entries, o.roster), c @ o.values))

    def objective(log_variances):
        state = PaceFilter(teams, drivers, VarianceComponents(*np.exp(log_variances)))
        total = 0.0
        for o, h, y in measurements:
            state.advance(o.event_index, o.season)
            state._ensure_entries(o.roster)
            state.mean, state.covariance, nll = gaussian_update(
                state.mean, state.covariance, h, y, getattr(state.variances, o.kind)
            )
            total += nll
        return total

    scale = max(float(np.median([np.var(o.values) for o in prior])), 0.01)
    starts = [
        scale * np.array([1, 0.2, 0.05, 0.005, 1, 0.5]),
        scale * np.array([0.3, 1, 0.2, 0.02, 0.5, 1]),
    ]
    bounds = [(np.log(1e-6), np.log(1e3))] * 6
    fitted = [
        minimize(
            objective,
            np.log(np.clip(s, 1e-6, 1e3)),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-8},
        )
        for s in starts
    ]
    valid = [f for f in fitted if f.success and np.isfinite(f.fun)]
    if not valid:
        raise RuntimeError(f"variance optimization failed: {[str(f.message) for f in fitted]}")
    best = min(valid, key=lambda f: f.fun)
    variances = VarianceComponents(*np.exp(best.x))
    return VarianceFit(
        variances,
        float(as_of_index),
        max(o.event_index for o in prior),
        len({o.race_id for o in prior}),
        float(best.fun),
        bool(best.success),
        tuple(
            {
                "converged": bool(f.success),
                "nll": float(f.fun),
                "iterations": int(f.nit),
                "message": str(f.message),
            }
            for f in fitted
        ),
        tuple(k for k, v in asdict(variances).items() if v <= 1.01e-6 or v >= 999.0),
    )


def filtered_state(
    observations: list[Observation], fit: VarianceFit, *, as_of_index: float
) -> PaceFilter:
    """Filter up to a prediction cutoff; never smooth using future outcomes."""
    if fit.as_of_index > as_of_index:
        raise ValueError("variance fit comes from the future")
    prior = [o for o in observations if o.event_index < as_of_index]
    roster = (
        pd.concat([o.roster for o in prior]).drop_duplicates()
        if prior
        else pd.DataFrame(columns=["constructor_id", "driver_id"])
    )
    state = PaceFilter(
        list(roster.constructor_id.unique()), list(roster.driver_id.unique()), fit.variances
    )
    for observation in prior:
        state.observe(observation)
    return state


def sample_ranking(
    entries: pd.DataFrame,
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    n_sims: int = 20000,
    seed: int = SEED,
    retirement: RetirementPosterior | None = None,
) -> pd.DataFrame:
    """Correlated Gaussian pace ranking with optional posterior retirements.

    A pace experiment cannot identify start/strategy/chaos variance separately.
    This sampler adds none. Retired entries follow finishers, ordered by sampled pace (an approximation
    to retirement order). Without a retirement posterior this is conditional
    on all entries finishing.
    """
    if not isinstance(n_sims, int) or n_sims <= 0 or len(entries) < 2:
        raise ValueError("positive simulation count and at least two entries required")
    if entries.driver_id.isna().any() or entries.driver_id.duplicated().any():
        raise ValueError("unique nonmissing drivers required")
    mean = np.asarray(mean, dtype=float)
    if mean.shape != (len(entries),) or not np.isfinite(mean).all():
        raise ValueError("invalid mean")
    cov = contrast_covariance(covariance)
    if cov.shape != (len(entries), len(entries)):
        raise ValueError("covariance does not match entries")
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if eigenvalues.min() < -1e-8:
        raise ValueError("covariance is not positive semidefinite")
    rng = np.random.default_rng(seed)
    draws = (
        mean
        + rng.standard_normal((n_sims, len(entries)))
        @ (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))).T
    )
    retired = (
        np.zeros_like(draws, dtype=bool)
        if retirement is None
        else retirement.sample(entries, n_sims, rng)
    )
    if retirement is not None:
        # A scored Grand Prix has a winner. Condition on at least one finisher,
        # rather than assigning a win/podium to a simulated retirement.
        for _ in range(1000):
            all_retired = retired.all(axis=1)
            if not all_retired.any():
                break
            retired[all_retired] = retirement.sample(entries, int(all_retired.sum()), rng)
        else:
            raise ValueError("unable to sample a race with at least one finisher")
    # Lexicographic sorting: classified first, then pace. No arbitrary offset.
    order = np.lexsort((-draws, retired), axis=1)
    positions = np.empty_like(order)
    positions[np.arange(n_sims)[:, None], order] = np.arange(1, len(entries) + 1)
    out = entries.copy()
    out["strength"] = mean
    counts = (~retired).sum(axis=0)
    conditional_finish = np.divide(
        np.where(retired, 0, positions).sum(axis=0),
        counts,
        out=np.full(len(entries), float(len(entries))),
        where=counts > 0,
    )
    out["expected_finish_if_classified"] = conditional_finish
    out["score"] = -conditional_finish
    out["win_probability"] = (positions == 1).mean(axis=0)
    out["podium_probability"] = ((positions <= 3) & ~retired).mean(axis=0)
    out["top10_probability"] = ((positions <= min(10, len(entries))) & ~retired).mean(axis=0)
    out["expected_finish"] = positions.mean(axis=0)
    out["finish_p10"], out["finish_p50"], out["finish_p90"] = np.quantile(
        positions, [0.1, 0.5, 0.9], axis=0, method="inverted_cdf"
    )
    out["dnf_probability_sim"] = retired.mean(axis=0)
    out["dnf_probability"] = (
        0.0 if retirement is None else retirement.probabilities(entries, at_least_one_finisher=True)
    )
    out["probability_scope"] = (
        "conditional_on_all_finishing" if retirement is None else "full_ranking"
    )
    out = out.sort_values(["score", "driver_id"], ascending=[False, True]).reset_index(drop=True)
    out["predicted_rank"] = np.arange(1, len(out) + 1)
    out["win_probability_rank"] = out.win_probability.rank(ascending=False, method="min").astype(
        int
    )
    return out
