"""Central configuration: paths, constants, seeds, and regulation regimes.

Every module reads immutable settings from here so experiments are reproducible
and no hidden constants live inside feature code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

ROOT: Final[Path] = Path(os.environ.get("F1PRED_HOME", Path.cwd())).resolve()

DATA_DIR: Final[Path] = ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
CACHE_DIR: Final[Path] = DATA_DIR / "cache"
RESULTS_DIR: Final[Path] = ROOT / "results"
IMAGES_DIR: Final[Path] = ROOT / "images"
MODELS_DIR: Final[Path] = ROOT / "models"
CONFIG_DIR: Final[Path] = ROOT / "configs"

SEED: Final[int] = 20260913

USER_AGENT: Final[str] = os.environ.get(
    "F1PRED_USER_AGENT", "f1pred (research; contact via repository)"
)

# Politeness budgets measured empirically against each API (see docs/BUILD_SPEC.md).
JOLPICA_MIN_INTERVAL_S: Final[float] = 0.30
# OpenF1 publishes two limits: at most 3 requests a second and 30 a minute. The
# spacing honours the first, the window the second; a spacing alone would allow
# 46 a minute and quietly breach the quota.
OPENF1_MIN_INTERVAL_S: Final[float] = 0.34
OPENF1_BURST: Final[tuple[int, float]] = (30, 60.0)
MULTIVIEWER_MIN_INTERVAL_S: Final[float] = 0.50
OPENMETEO_MIN_INTERVAL_S: Final[float] = 0.50

HTTP_TIMEOUT_S: Final[int] = 30
HTTP_RETRIES: Final[int] = 4


@dataclass(frozen=True)
class Regime:
    """A technical-regulation era. Performance is not comparable across regimes."""

    name: str
    first_season: int
    last_season: int | None  # None == open ended

    def contains(self, season: int) -> bool:
        return season >= self.first_season and (
            self.last_season is None or season <= self.last_season
        )


REGIMES: Final[tuple[Regime, ...]] = (
    Regime("V6_HYBRID_2014", 2014, 2016),
    Regime("WIDE_AERO_2017", 2017, 2021),
    Regime("GROUND_EFFECT_2022", 2022, 2025),
    Regime("NEW_PU_AERO_2026", 2026, None),
)


def regime_for(season: int) -> str:
    """Return the regulation-regime label for a season."""
    for r in REGIMES:
        if r.contains(season):
            return r.name
    return "UNKNOWN"


# Prediction snapshots, ordered. A feature may only be used at or after its stage.
STAGES: Final[tuple[str, ...]] = (
    "PRE_WEEKEND",
    "POST_FP1",
    "POST_FP2",
    "POST_FP3",
    "POST_QUALI",
    "RACE_START",
)
STAGE_ORDER: Final[dict[str, int]] = {s: i for i, s in enumerate(STAGES)}
# POST_RACE is deliberately outside STAGES: it can never be a prediction stage.
POST_RACE: Final[str] = "POST_RACE"


@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters for the strength/pace model. Tuned on pre-2026 folds only."""

    # Half-life in races for exponential recency weighting of observations.
    # Selected on the 2022-2025 walk-forward window (92 races); the objective
    # surface is flat between 3 and 9 races, so 6 is a plateau choice, not a peak.
    recency_half_life_races: float = 6.0
    # Extra multiplicative down-weight applied to observations from a previous
    # regime. Also selected on 2022-2025; 2026 was never used for tuning.
    cross_regime_weight: float = 0.35
    # Ridge strength for the driver/team fixed-effects decomposition.
    ridge_alpha: float = 1.0
    # Hand-set shrinkage counts pulling sparse driver coefficients toward zero.
    driver_shrinkage_races: float = 3.0
    # Beta prior for DNF rates: alpha failures / (alpha+beta) pseudo-races.
    dnf_prior_alpha: float = 1.2
    dnf_prior_beta: float = 12.0
    n_simulations: int = 50_000
    seed: int = SEED
    #: Where the simulator's per-entry strength spread comes from. "posterior" is
    #: the ridge posterior standard error of team+driver including their
    #: covariance; "heuristic" is the inherited 0.45/(1 + weight/3) formula.
    #: The posterior arm discards cross-entry covariance and is diagnostic only.
    strength_spread: str = "heuristic"


DEFAULT_MODEL_CONFIG: Final[ModelConfig] = ModelConfig()

# Archived variance fit on 2024-2025 (48 races). Its marginal-uncertainty input
# was misspecified for rankings, and the 2022-2023 evaluation reversed time.
# These values are retained for provenance, not as a validated variance budget.
# See results/simulator_calibration/ and scripts/calibrate_simulator.py.
FITTED_VARIANCE_BUDGET: Final[dict[str, float]] = {
    "qualifying_noise": 0.0784,
    "race_pace_noise": 0.2071,
    "start_swing_slots": 0.1468,
    "strategy_swing_slots": 0.5496,
}


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to."""
    for p in (RAW_DIR, PROCESSED_DIR, CACHE_DIR, RESULTS_DIR, IMAGES_DIR, MODELS_DIR):
        p.mkdir(parents=True, exist_ok=True)
