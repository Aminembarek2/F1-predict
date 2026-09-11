"""Experimental team retirement rates with a fitted beta-binomial hierarchy.

A team shares one latent failure rate across its entries. Its posterior is
integrated when sampling. This represents uncertainty about reliability, not a
claim to model race-wide incidents or an identifiable driver failure effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, expit


@dataclass(frozen=True)
class RetirementPosterior:
    alpha: float
    beta: float
    teams: dict[str, tuple[float, float]]
    last_training_index: float
    converged: bool
    at_boundary: bool

    def sample(self, entries: pd.DataFrame, n: int, rng: np.random.Generator) -> np.ndarray:
        retired = np.zeros((n, len(entries)), dtype=bool)
        for team, indices in (
            entries.reset_index(drop=True).groupby("constructor_id").groups.items()
        ):
            a, b = self.teams.get(team, (self.alpha, self.beta))
            probability = rng.beta(a, b, size=(n, 1))
            retired[:, indices] = rng.random((n, len(indices))) < probability
        return retired

    def probabilities(
        self, entries: pd.DataFrame, *, at_least_one_finisher: bool = False
    ) -> np.ndarray:
        probabilities = np.array(
            [
                a / (a + b)
                for a, b in [
                    self.teams.get(t, (self.alpha, self.beta)) for t in entries.constructor_id
                ]
            ]
        )
        if at_least_one_finisher:
            log_all_retired = 0.0
            for team, group in entries.groupby("constructor_id"):
                a, b = self.teams.get(team, (self.alpha, self.beta))
                log_all_retired += betaln(a + len(group), b) - betaln(a, b)
            all_retired = np.exp(log_all_retired)
            if all_retired >= 1:
                raise ValueError("no finite probability of a finisher")
            probabilities = (probabilities - all_retired) / (1 - all_retired)
        return probabilities


def fit_retirement(
    results: pd.DataFrame, *, as_of_index: float, target_regime: str
) -> RetirementPosterior:
    """Fit pooling strength on past constructor-by-regime counts, then update.

    Previous regimes learn the population prior but not a current team's count.
    At a new regime every team starts with that learned population prior.
    """
    prior = results.loc[(results.event_index < as_of_index) & results.started.eq(1)].copy()
    if prior.empty or not prior.dnf.isin([0, 1]).all():
        raise ValueError("valid earlier starter outcomes required")
    counts = prior.groupby(["regime", "constructor_id"]).dnf.agg(["sum", "count"])
    failed, total = counts["sum"].to_numpy(float), counts["count"].to_numpy(float)
    if len(counts) < 3:
        raise ValueError("at least three historical constructor/regime groups required")

    def objective(parameters):
        mu, k = expit(parameters[0]), np.exp(parameters[1])
        a, b = mu * k, (1 - mu) * k
        return -np.sum(betaln(a + failed, b + total - failed) - betaln(a, b))

    rate = np.clip(failed.sum() / total.sum(), 1e-5, 1 - 1e-5)
    bounds = [(-12, 12), (np.log(1e-3), np.log(1e6))]
    fits = [
        minimize(
            objective, [np.log(rate / (1 - rate)), np.log(k)], method="L-BFGS-B", bounds=bounds
        )
        for k in (10, 100)
    ]
    valid = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not valid:
        raise RuntimeError("retirement hierarchy failed to converge")
    fit = min(valid, key=lambda x: x.fun)
    mu, k = expit(fit.x[0]), np.exp(fit.x[1])
    a, b = float(mu * k), float((1 - mu) * k)
    current = (
        prior.loc[prior.regime.eq(target_regime)]
        .groupby("constructor_id")
        .dnf.agg(["sum", "count"])
    )
    teams = {t: (a + row["sum"], b + row["count"] - row["sum"]) for t, row in current.iterrows()}
    return RetirementPosterior(
        a,
        b,
        teams,
        float(prior.event_index.max()),
        True,
        bool(
            any(
                abs(x - lo) < 1e-3 or abs(x - hi) < 1e-3
                for x, (lo, hi) in zip(fit.x, bounds, strict=True)
            )
        ),
    )
