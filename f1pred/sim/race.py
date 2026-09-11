"""Monte Carlo race simulator.

The simulator is deliberately transparent: every random draw corresponds to a
named physical source of race-day variance, so a forecast can be decomposed into
"how much of this probability comes from pace, from the start, from strategy,
from retirements".

Simulation of one race
----------------------
1. **Qualifying** (only when the grid is unknown, i.e. stages before
   ``POST_QUALI``): each entry draws a one-lap performance from its latent
   strength plus single-lap noise; the ordering becomes the grid.
2. **Race pace**: an independent draw from latent strength plus race-pace noise.
3. **Track position**: the finishing order blends grid and race pace. The blend
   weight is the circuit's *overtaking difficulty* - at Monaco the grid almost
   fully determines the result, at Monza pace dominates. This is the mechanism
   that makes "P5 at Monaco" different from "P5 at Monza".
4. **Retirements**: an independent Bernoulli draw per entry, with a shared
   race-level shock so that chaotic races retire several cars at once.
5. Retired cars are classified behind every finisher, ordered by the lap they
   reached (approximated by their sampled performance).

All randomness flows from one seeded ``numpy`` generator, so results are exactly
reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import SEED


@dataclass(frozen=True)
class SimulationSettings:
    """Variance budget for the simulator, in latent-strength units.

    These are the inherited constants, and they are still an incumbent rather
    than a validated optimum. A maximum-likelihood fit of the four performance
    terms was run and **rejected**: refitted against a measured strength
    posterior it looked like a gain, but tested against the real control it cost
    0.219 nats of winner log-loss. The fit and the rejection are in
    ``docs/FINDINGS.md`` and ``results/architecture_experiment/``.

    That they were never fitted remains the honest criticism of this simulator.
    It is not resolved by replacing a chosen number with a worse fitted one.
    """

    n_sims: int = 50_000
    seed: int = SEED
    #: Spread of one-lap performance around latent strength (drives grid uncertainty).
    qualifying_noise: float = 0.55
    #: Spread of race pace around latent strength.
    race_pace_noise: float = 0.60
    #: Extra spread for drivers with little observation history (rookies, new teams).
    rookie_extra_noise: float = 0.45
    #: Standard deviation of first-lap / start-phase position swings, in grid slots.
    start_swing_slots: float = 1.6
    #: Strategy and safety-car luck, in grid slots.
    strategy_swing_slots: float = 2.0
    #: Probability that a race is "chaotic" (multiple retirements, safety cars).
    chaos_probability: float = 0.22
    #: DNF hazard multiplier applied in a chaotic race.
    chaos_dnf_multiplier: float = 2.1

    def __post_init__(self) -> None:
        if not isinstance(self.n_sims, int) or self.n_sims <= 0:
            raise ValueError("n_sims must be a positive integer")
        if not 0 <= self.chaos_probability <= 1 or self.chaos_dnf_multiplier < 1:
            raise ValueError("invalid chaos settings")
        for name in (
            "qualifying_noise",
            "race_pace_noise",
            "rookie_extra_noise",
            "start_swing_slots",
            "strategy_swing_slots",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def conditional_dnf_rates(
    probabilities: np.ndarray, chaos_probability: float, multiplier: float
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve marginal DNF probabilities while correlating attrition by race."""
    p = np.asarray(probabilities, dtype=float)
    q, m = float(chaos_probability), float(multiplier)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("DNF probabilities must be finite and in [0, 1]")
    if not 0 <= q <= 1 or not np.isfinite(m) or m < 1:
        raise ValueError("invalid chaos settings")
    if q == 1:
        return p / m, p
    ordinary = p / (1 + q * (m - 1))
    ordinary = np.where(ordinary * m <= 1, ordinary, (p - q) / (1 - q))
    return ordinary, np.minimum(ordinary * m, 1.0)


def position_to_latent(positions: np.ndarray, field_size: int) -> np.ndarray:
    """Map 1-based positions onto the latent-strength scale (higher is better).

    Identical to :func:`f1pred.features.pace.normal_scores` but vectorised over a
    plain array, so grid slots and fitted strengths share one set of units.
    """
    from scipy.stats import norm

    quantile = (np.asarray(positions, dtype=float) - 0.5) / float(field_size)
    return -norm.ppf(np.clip(quantile, 1e-4, 1 - 1e-4))


def overtaking_blend(overtaking_difficulty: float) -> float:
    """Weight given to grid position when forming the finishing order.

    ``overtaking_difficulty`` is on 0 (trivially easy to pass, e.g. Monza-like)
    to 1 (essentially impossible, e.g. Monaco). The returned weight is the share
    of the finishing-order score contributed by the starting slot.
    """
    d = float(np.clip(overtaking_difficulty, 0.0, 1.0))
    # Even at the easiest circuits track position is worth something, and even at
    # the hardest a much faster car eventually gets through.
    return 0.18 + 0.70 * d


@dataclass
class RaceForecast:
    """Output of a simulated race."""

    table: pd.DataFrame
    settings: SimulationSettings
    grid_known: bool
    diagnostics: dict[str, float] = field(default_factory=dict)


def simulate_race(
    entries: pd.DataFrame,
    *,
    overtaking_difficulty: float,
    settings: SimulationSettings | None = None,
    strength_col: str = "strength",
    dnf_col: str = "dnf_probability",
    grid_col: str | None = None,
    strength_sd_col: str | None = "strength_sd",
    qualifying_strength_col: str | None = None,
) -> RaceForecast:
    """Simulate a race many times and return outcome probabilities.

    Parameters
    ----------
    entries:
        One row per starter. Must contain ``driver_id``, ``constructor_id``,
        ``strength_col`` and ``dnf_col``.
    overtaking_difficulty:
        0-1 circuit property; see :func:`overtaking_blend`.
    grid_col:
        Name of a known-grid column. When ``None`` the grid is itself simulated,
        which is the correct behaviour before qualifying has run.
    strength_sd_col:
        Per-entry extra pace uncertainty (e.g. rookies, new cars). Optional.

    Returns
    -------
    RaceForecast
        ``table`` carries win/podium/top-10/DNF probabilities, expected finish
        and the full finishing distribution percentiles.
    """
    settings = settings or SimulationSettings()
    df = entries.reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        raise ValueError("no entries to simulate")
    if df["driver_id"].isna().any() or df["driver_id"].duplicated().any():
        raise ValueError("drivers must be unique and nonmissing")
    rng = np.random.default_rng(settings.seed)

    strength = pd.to_numeric(df[strength_col], errors="coerce").fillna(0.0).to_numpy(float)
    # Qualifying is a different competition from the race: a car can be sharp over
    # one lap and ordinary in race trim. When a separate one-lap strength is
    # supplied it drives the grid draw; otherwise the single strength does both.
    qual_strength = (
        pd.to_numeric(df[qualifying_strength_col], errors="coerce").fillna(0.0).to_numpy(float)
        if qualifying_strength_col and qualifying_strength_col in df
        else strength
    )
    dnf_p = pd.to_numeric(df[dnf_col], errors="coerce").to_numpy(float)
    ordinary_hazard, chaos_hazard = conditional_dnf_rates(
        dnf_p, settings.chaos_probability, settings.chaos_dnf_multiplier
    )
    extra_sd = (
        pd.to_numeric(df[strength_sd_col], errors="coerce").fillna(0.0).to_numpy(float)
        if strength_sd_col and strength_sd_col in df
        else np.zeros(n)
    )

    qual_sd = np.sqrt(settings.qualifying_noise**2 + extra_sd**2)
    race_sd = np.sqrt(settings.race_pace_noise**2 + extra_sd**2)

    grid_known = grid_col is not None and grid_col in df
    if grid_col is not None and not grid_known:
        raise ValueError(f"missing known-grid column {grid_col}")
    if grid_known:
        # Grid slots are re-ranked into a dense 1..n starting order. Raw slot
        # numbers can exceed the number of starters when cars are withdrawn after
        # qualifying, and a pit-lane start is recorded as 0; what the simulator
        # needs is the order cars line up in, not the nominal slot.
        raw = pd.to_numeric(df[grid_col], errors="coerce")
        # A pit-lane start (0) is the back of the field, not the front.
        raw = raw.replace(0, np.nan).fillna(raw.max() + 1 if raw.notna().any() else 1)
        fixed_grid = raw.rank(method="first").to_numpy(float)

    w_grid = overtaking_blend(overtaking_difficulty)
    w_pace = 1.0 - w_grid
    # Positions are converted to the latent-strength scale with the same
    # van-der-Waerden transform used to score race results, so a grid slot and a
    # pace estimate are directly comparable and the blend weight means what it says.
    position_scale = position_to_latent(np.arange(1, n + 1), n)
    # Value of one grid slot in latent units, used to price start/strategy swings.
    slot_value = float(position_scale[0] - position_scale[-1]) / max(n - 1, 1)

    sims = int(settings.n_sims)
    wins = np.zeros(n)
    podiums = np.zeros(n)
    top10s = np.zeros(n)
    dnfs = np.zeros(n)
    pole = np.zeros(n)
    finish_sum = np.zeros(n)
    classified_sum = np.zeros(n)
    classified_count = np.zeros(n)
    finish_hist = np.zeros((n, n), dtype=np.int64)

    batch = 2000
    done = 0
    while done < sims:
        b = min(batch, sims - done)

        # 1. Grid.
        if grid_known:
            grid = np.tile(fixed_grid, (b, 1))
        else:
            qperf = qual_strength + rng.normal(0.0, qual_sd, size=(b, n))
            grid = (np.argsort(np.argsort(-qperf, axis=1), axis=1) + 1).astype(float)
            pole += (grid == 1).sum(axis=0)

        # 2. Race pace.
        pace = strength + rng.normal(0.0, race_sd, size=(b, n))

        # 3. Track position: blend the grid advantage with race pace, then add
        #    start-phase and strategy/safety-car swings priced in grid slots.
        grid_advantage = position_scale[grid.astype(np.int64) - 1]
        swing = rng.normal(0.0, settings.start_swing_slots * slot_value, size=(b, n))
        strat = rng.normal(0.0, settings.strategy_swing_slots * slot_value, size=(b, n))
        score = w_grid * (grid_advantage + swing + strat) + w_pace * pace

        # 4. Retirements, with a correlated race-level chaos shock.
        chaotic = rng.random((b, 1)) < settings.chaos_probability
        hazard = np.where(chaotic, chaos_hazard, ordinary_hazard)
        retired = rng.random((b, n)) < hazard

        # 5. Classify: finishers first by score, retirements behind, worst pace last.
        final_score = np.where(retired, score - 1e6, score)
        order = np.argsort(-final_score, axis=1)
        positions = np.empty((b, n), dtype=np.int64)
        rows = np.arange(b)[:, None]
        positions[rows, order] = np.arange(1, n + 1)[None, :]

        wins += (positions == 1).sum(axis=0)
        podiums += (positions <= 3).sum(axis=0)
        top10s += (positions <= min(10, n)).sum(axis=0)
        dnfs += retired.sum(axis=0)
        finish_sum += positions.sum(axis=0)
        # Expected position *given the car finishes* separates pace from attrition.
        classified_sum += np.where(retired, 0, positions).sum(axis=0)
        classified_count += (~retired).sum(axis=0)
        for pos in range(1, n + 1):
            finish_hist[:, pos - 1] += (positions == pos).sum(axis=0)
        done += b

    out = df.copy()
    out["win_probability"] = wins / sims
    out["podium_probability"] = podiums / sims
    out["top10_probability"] = top10s / sims
    out["dnf_probability_sim"] = dnfs / sims
    out["expected_finish"] = finish_sum / sims
    out["expected_finish_if_classified"] = np.where(
        classified_count > 0, classified_sum / np.maximum(classified_count, 1), float(n)
    )
    if not grid_known:
        out["pole_probability"] = pole / sims
    cum = finish_hist.cumsum(axis=1) / sims
    out["finish_p10"] = [int(np.searchsorted(cum[i], 0.10) + 1) for i in range(n)]
    out["finish_p50"] = [int(np.searchsorted(cum[i], 0.50) + 1) for i in range(n)]
    out["finish_p90"] = [int(np.searchsorted(cum[i], 0.90) + 1) for i in range(n)]
    # Rank on expected classified position: it is continuous (so it cannot tie the
    # way win probability does deep in the field) and it is not dominated by the
    # retirement tail the way the unconditional expected finish is.
    out["pace_rank_score"] = -out["expected_finish_if_classified"]
    out = out.sort_values(
        ["win_probability", "pace_rank_score"], ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    out["win_probability_rank"] = np.arange(1, n + 1)
    out = out.sort_values(
        ["pace_rank_score", "driver_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    out["predicted_rank"] = np.arange(1, n + 1)

    diagnostics = {
        "used_separate_qualifying_strength": float(
            bool(qualifying_strength_col and qualifying_strength_col in df)
        ),
        "win_probability_sum": float(out["win_probability"].sum()),
        "podium_probability_sum": float(out["podium_probability"].sum()),
        "grid_weight": w_grid,
        "n_sims": float(sims),
    }
    return RaceForecast(
        table=out, settings=settings, grid_known=grid_known, diagnostics=diagnostics
    )
