"""DNF / reliability modelling, kept separate from pace.

Retirements were the largest single source of ranking error in the original notebook,
so failure probability is modelled explicitly rather than folded into pace.

Data limitation that shapes this design
---------------------------------------
Classic Ergast attributed a cause to every retirement ("Engine", "Collision",
...). From 2024 onward the Jolpica feed reports a **generic "Retired"** for
essentially all DNFs, so a mechanical-versus-incident split is not identifiable
in the current regime (verified in ``results/data_quality.csv``). This module
therefore models the *total* DNF hazard, decomposed hierarchically into:

``team hazard``
    Shared by both cars of a constructor - captures car and power-unit reliability.
``driver hazard``
    The residual on top of the team, identified against the team-mate - captures
    crash-proneness and race-craft risk.
``circuit multiplier``
    How attritional a venue is relative to the calendar average. Unknown venues
    fall back to a *track-type* prior (street circuits are markedly more
    attritional than permanent ones), which is what Madrid 2026 requires.

Rates are Beta-smoothed and recency-weighted, and every aggregate is computed
from races strictly before the target race.

Availability: ``PRE_WEEKEND``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import DEFAULT_MODEL_CONFIG, ModelConfig
from .pace import recency_weights

EPS = 1e-9


def beta_rate(events: float, trials: float, alpha: float, beta: float) -> float:
    """Posterior mean of a Beta-Binomial rate."""
    return float((events + alpha) / (trials + alpha + beta))


@dataclass(frozen=True)
class ReliabilityEstimate:
    """Fitted DNF hazards at one point in time."""

    teams: pd.DataFrame  # constructor_id, team_dnf_rate, weight
    drivers: pd.DataFrame  # driver_id, driver_dnf_rate, driver_excess, weight
    base_rate: float
    n_observations: int

    def hazard(self, driver_id: str, constructor_id: str, *, multiplier: float = 1.0) -> float:
        """DNF probability for one entry at a venue with the given multiplier."""
        team = self.teams.set_index("constructor_id")["team_dnf_rate"]
        excess = self.drivers.set_index("driver_id")["driver_excess"]
        base = float(team.get(constructor_id, self.base_rate))
        # Driver excess is additive in log-odds so it cannot push a rate past 0 or 1.
        odds = np.log(np.clip(base, EPS, 1 - EPS) / (1 - np.clip(base, EPS, 1 - EPS)))
        odds += float(excess.get(driver_id, 0.0))
        odds += np.log(max(multiplier, EPS))
        p = 1.0 / (1.0 + np.exp(-odds))
        return float(np.clip(p, 0.01, 0.60))


def fit_reliability(
    results: pd.DataFrame,
    *,
    as_of_index: float,
    target_regime: str,
    config: ModelConfig = DEFAULT_MODEL_CONFIG,
) -> ReliabilityEstimate:
    """Estimate team and driver DNF hazards from races before ``as_of_index``.

    Parameters
    ----------
    results:
        Starter rows only, carrying ``event_index``, ``dnf``, ``driver_id``,
        ``constructor_id`` and ``regime``.
    as_of_index:
        Chronological index of the target race; observations must precede it.
    target_regime:
        Prior-regime observations are down-weighted by ``cross_regime_weight``.
        Reliability changes sharply at a regulation break, so this matters more
        here than for pace.
    """
    obs = results.loc[(results["event_index"] < as_of_index) & (results["started"] == 1)].copy()
    if obs.empty:
        empty_t = pd.DataFrame(columns=["constructor_id", "team_dnf_rate", "weight"])
        empty_d = pd.DataFrame(columns=["driver_id", "driver_dnf_rate", "driver_excess", "weight"])
        return ReliabilityEstimate(empty_t, empty_d, 0.12, 0)

    w = recency_weights(obs["event_index"], as_of_index, config.recency_half_life_races * 2.0)
    w = w * np.where(obs["regime"].eq(target_regime), 1.0, config.cross_regime_weight)
    obs["w"] = w.astype(float)
    obs = obs.loc[obs["w"] > 1e-6]
    if obs.empty:
        empty_t = pd.DataFrame(columns=["constructor_id", "team_dnf_rate", "weight"])
        empty_d = pd.DataFrame(columns=["driver_id", "driver_dnf_rate", "driver_excess", "weight"])
        return ReliabilityEstimate(empty_t, empty_d, 0.12, 0)

    base_rate = float(np.average(obs["dnf"].to_numpy(float), weights=obs["w"].to_numpy(float)))
    a, b = config.dnf_prior_alpha, config.dnf_prior_beta

    team = obs.groupby("constructor_id").apply(
        lambda g: pd.Series(
            {
                "events": float((g["dnf"] * g["w"]).sum()),
                "trials": float(g["w"].sum()),
            }
        ),
        include_groups=False,
    )
    # Shrink each team toward the field base rate with the Beta prior.
    team["team_dnf_rate"] = (team["events"] + (a + b) * base_rate) / (team["trials"] + a + b)
    team["weight"] = team["trials"]
    teams = team.reset_index()[["constructor_id", "team_dnf_rate", "weight"]]

    # Driver excess hazard, identified against the car the driver actually drove.
    rate_map = teams.set_index("constructor_id")["team_dnf_rate"]
    obs["team_rate"] = obs["constructor_id"].map(rate_map).fillna(base_rate)
    drv = obs.groupby("driver_id").apply(
        lambda g: pd.Series(
            {
                "events": float((g["dnf"] * g["w"]).sum()),
                "trials": float(g["w"].sum()),
                "expected": float((g["team_rate"] * g["w"]).sum()),
            }
        ),
        include_groups=False,
    )
    k = config.driver_shrinkage_races * 2.0
    drv["driver_dnf_rate"] = (
        drv["events"] + k * (drv["expected"] / drv["trials"].clip(lower=EPS))
    ) / (drv["trials"] + k)
    expected_rate = (drv["expected"] / drv["trials"].clip(lower=EPS)).clip(EPS, 1 - EPS)
    observed_rate = drv["driver_dnf_rate"].clip(EPS, 1 - EPS)
    drv["driver_excess"] = np.log(observed_rate / (1 - observed_rate)) - np.log(
        expected_rate / (1 - expected_rate)
    )
    drv["weight"] = drv["trials"]
    drivers = drv.reset_index()[["driver_id", "driver_dnf_rate", "driver_excess", "weight"]]

    return ReliabilityEstimate(
        teams=teams, drivers=drivers, base_rate=base_rate, n_observations=len(obs)
    )


#: Circuit layout families. Street circuits are walled with no run-off, which
#: raises incident attrition; permanent circuits with wide run-off lower it.
STREET_CIRCUITS = frozenset(
    {
        "monaco",
        "baku",
        "marina_bay",
        "vegas",
        "jeddah",
        "miami",
        "albert_park",
        "villeneuve",
        "madring",
        "valencia",
        "sochi",
        "detroit",
        "adelaide",
    }
)


def circuit_dnf_multipliers(results: pd.DataFrame, *, pseudo_entries: float = 80.0) -> pd.DataFrame:
    """Per-circuit attrition multipliers relative to the calendar base rate.

    Shrunk toward 1.0 with ``pseudo_entries`` of prior mass so a circuit with two
    visits cannot claim an extreme multiplier.
    """
    starters = results.loc[results["started"] == 1]
    base = float(starters["dnf"].mean())
    grouped = starters.groupby("circuit_id").agg(entries=("dnf", "size"), dnfs=("dnf", "sum"))
    grouped["circuit_dnf_rate"] = (grouped["dnfs"] + pseudo_entries * base) / (
        grouped["entries"] + pseudo_entries
    )
    grouped["dnf_multiplier"] = grouped["circuit_dnf_rate"] / max(base, EPS)
    grouped["is_street"] = grouped.index.isin(STREET_CIRCUITS)
    return grouped.reset_index()


def track_type_multiplier(results: pd.DataFrame, *, is_street: bool) -> float:
    """Attrition multiplier for a circuit with no history, from its layout family.

    This is the fallback Madrid 2026 needs: a brand-new street circuit has no
    visits, so the multiplier comes from how street circuits behave on average.
    """
    starters = results.loc[results["started"] == 1].copy()
    starters["is_street"] = starters["circuit_id"].isin(STREET_CIRCUITS)
    base = float(starters["dnf"].mean())
    subset = starters.loc[starters["is_street"] == is_street, "dnf"]
    if len(subset) < 100:
        return 1.0
    return float(subset.mean() / max(base, EPS))
