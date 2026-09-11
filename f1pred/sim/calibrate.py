"""Fit the simulator's variance budget instead of choosing it.

Seven constants define everything probabilistic the simulator does - how much a
qualifying lap scatters around latent pace, how much race pace scatters, how big
a start or a strategy call swings the order, and how retirements cluster. In the
inherited implementation all seven were hand-set and never fitted to anything.

That has a consequence worth stating plainly: a Monte Carlo whose noise is chosen
rather than estimated is a softmax with a hand-picked temperature wearing a
costume, which is exactly what the candidate experiment found when a
one-parameter softmax tied it on winner log-loss.

This fits them by maximum likelihood on the tuning window, using the same
walk-forward discipline as everything else: each race is scored by a model that
saw only earlier races. Held-out seasons are never touched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..config import SEED
from .race import SimulationSettings, simulate_race

log = logging.getLogger(__name__)

EPS = 1e-12

#: Parameters that are fitted, with the interval each is constrained to. The
#: chaos pair is excluded: those two are pinned by the retirement marginals the
#: forecast publishes, and refitting them here would quietly change a contract
#: the rest of the system relies on.
BOUNDS: dict[str, tuple[float, float]] = {
    "qualifying_noise": (0.05, 2.00),
    "race_pace_noise": (0.05, 2.00),
    "start_swing_slots": (0.00, 6.00),
    "strategy_swing_slots": (0.00, 6.00),
}


@dataclass(frozen=True)
class CalibrationResult:
    """Fitted variance budget, and what it was worth against the incumbent."""

    settings: SimulationSettings
    incumbent: SimulationSettings
    fitted_logloss: float
    incumbent_logloss: float
    n_races: int
    n_evaluations: int
    trace: pd.DataFrame

    @property
    def improvement(self) -> float:
        """Reduction in mean winner log-loss. Positive means the fit is better."""
        return self.incumbent_logloss - self.fitted_logloss


def _to_free(values: np.ndarray) -> np.ndarray:
    """Map bounded parameters onto the whole line, so the search is unconstrained."""
    lo = np.array([b[0] for b in BOUNDS.values()])
    hi = np.array([b[1] for b in BOUNDS.values()])
    unit = np.clip((values - lo) / (hi - lo), 1e-6, 1 - 1e-6)
    return np.log(unit / (1 - unit))


def _from_free(free: np.ndarray) -> np.ndarray:
    lo = np.array([b[0] for b in BOUNDS.values()])
    hi = np.array([b[1] for b in BOUNDS.values()])
    return lo + (hi - lo) / (1.0 + np.exp(-free))


def winner_logloss(
    races: list[tuple[pd.DataFrame, float, np.ndarray]],
    settings: SimulationSettings,
) -> float:
    """Mean negative log-probability assigned to the driver who actually won.

    A proper score: it rewards a forecast for being confident *and* right, and
    punishes confident and wrong, which is what a variance budget controls.
    """
    total = 0.0
    for scored, difficulty, winner_mask in races:
        sim = simulate_race(scored, overtaking_difficulty=difficulty, settings=settings)
        p = sim.table.set_index("driver_id")["win_probability"]
        p = p.reindex(scored["driver_id"]).to_numpy(float)
        total -= np.log(max(float(p[winner_mask].sum()), EPS))
    return total / max(len(races), 1)


def fit(
    races: list[tuple[pd.DataFrame, float, np.ndarray]],
    *,
    incumbent: SimulationSettings | None = None,
    n_sims: int = 4000,
    max_iterations: int = 120,
) -> CalibrationResult:
    """Maximum-likelihood fit of the variance budget on the supplied races.

    ``races`` holds one tuple per race: the scored entry table the simulator
    consumes, the circuit's overtaking difficulty, and a boolean mask marking the
    actual winner. Every one of those is built walk-forward by the caller.
    """
    if not races:
        raise ValueError("no races supplied to calibrate against")
    incumbent = incumbent or SimulationSettings()
    # One fixed seed across every evaluation, so the optimiser compares parameter
    # sets rather than chasing simulation noise between them.
    base = replace(incumbent, n_sims=n_sims, seed=SEED)

    history: list[dict] = []

    def objective(free: np.ndarray) -> float:
        values = _from_free(free)
        trial = replace(base, **dict(zip(BOUNDS, values, strict=True)))
        loss = winner_logloss(races, trial)
        history.append({**dict(zip(BOUNDS, values, strict=True)), "winner_logloss": loss})
        if len(history) % 20 == 0:
            log.info("evaluation %d: log-loss %.4f", len(history), loss)
        return loss

    start = np.array([getattr(incumbent, name) for name in BOUNDS], dtype=float)
    result = minimize(
        objective,
        _to_free(start),
        method="Nelder-Mead",
        options={"maxfev": max_iterations, "xatol": 0.02, "fatol": 1e-4},
    )
    fitted_values = _from_free(result.x)
    fitted = replace(base, **dict(zip(BOUNDS, fitted_values, strict=True)))

    return CalibrationResult(
        settings=fitted,
        incumbent=base,
        fitted_logloss=float(winner_logloss(races, fitted)),
        incumbent_logloss=float(winner_logloss(races, base)),
        n_races=len(races),
        n_evaluations=len(history),
        trace=pd.DataFrame(history),
    )
