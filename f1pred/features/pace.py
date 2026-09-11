"""Latent pace-strength estimation from historical race and qualifying outcomes.

Why not rolling finishing position
----------------------------------
Finishing position is a censored, track-position-dominated signal: a fast car
stuck behind a slow one scores the slow car's result, and a retirement scores
last regardless of pace. The 2025 baseline showed rolling-finish form reaching
only ~25% winner Top-1. This module instead estimates a *latent pace* on a
continuous, cross-circuit-comparable scale:

* qualifying observations use percentage gap to session pole (clean single-lap
  pace, unaffected by strategy or traffic);
* race observations use a normal-score transform of classified finishing order,
  with retirements excluded rather than scored as last.

Both are then fitted with additive team and driver indicators using weighted ridge
regression. Equal penalties select a confounded decomposition; these are not
independently identified driver skills. The fit uses with exponential recency weighting
and an explicit down-weight across regulation regimes.

Availability: ``PRE_WEEKEND`` - every observation comes from races strictly
before the target race.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import DEFAULT_MODEL_CONFIG, ModelConfig, regime_for

EPS = 1e-9


def normal_scores(positions: pd.Series, field_size: pd.Series) -> pd.Series:
    """Map finishing positions to a latent-normal scale (van der Waerden scores).

    Position 1 in a 20-car field maps to roughly +1.67 sigma, the median to 0.
    The sign is flipped so that **higher is better**, matching every other
    strength quantity in the project.

    Parameters
    ----------
    positions:
        1-based classified finishing positions. ``NaN`` propagates.
    field_size:
        Number of classified runners in the same race.
    """
    p = pd.to_numeric(positions, errors="coerce")
    n = pd.to_numeric(field_size, errors="coerce")
    quantile = (p - 0.5) / n
    return pd.Series(-norm.ppf(quantile.clip(1e-4, 1 - 1e-4)), index=p.index).where(p.notna())


def qualifying_pace_score(gap_pct: pd.Series) -> pd.Series:
    """Convert percentage gap-to-pole into a higher-is-better standardised score.

    A 1% qualifying deficit is roughly 0.9 s at a 90 s circuit. Dividing by a
    fixed 0.75% scale keeps the units comparable to :func:`normal_scores`
    (both land in roughly +-2 for a normal field) without fitting anything to
    the evaluation data.
    """
    g = pd.to_numeric(gap_pct, errors="coerce")
    return -g / 0.75


@dataclass(frozen=True)
class StrengthEstimate:
    """Latent strengths at one point in time."""

    drivers: pd.DataFrame  # driver_id, driver_effect, n_obs, shrunk_effect, effect_se
    teams: pd.DataFrame  # constructor_id, team_effect, n_obs, effect_se
    as_of: str
    n_observations: int
    intercept: float
    #: Residual standard deviation of the weighted fit, in latent units.
    residual_sd: float = 0.0
    #: Posterior covariance of [intercept, teams..., drivers...] and the row each
    #: name occupies. Kept whole because team and driver effects are confounded:
    #: each has a wide marginal interval and they are strongly negatively
    #: correlated, so only the covariance gives the right width for their *sum*.
    covariance: np.ndarray | None = None
    team_rows: dict[str, int] = field(default_factory=dict)
    driver_rows: dict[str, int] = field(default_factory=dict)

    def strength_variance(self, driver_id: str, constructor_id: str) -> float:
        """Posterior variance of team + driver, including their covariance."""
        if self.covariance is None:
            return float("nan")
        c = self.covariance
        t = self.team_rows.get(constructor_id)
        d = self.driver_rows.get(driver_id)
        if t is None and d is None:
            return float(np.max(np.diag(c)))
        if t is None or d is None:
            row = t if t is not None else d
            # One side unseen: its own spread, widened by the worst the fit saw,
            # so "never observed" cannot read as "confidently average".
            return float(c[row, row] + np.max(np.diag(c)))
        return float(c[t, t] + c[d, d] + 2.0 * c[t, d])

    def driver_strength(self, driver_id: str, constructor_id: str) -> float:
        """Combined team + driver strength, falling back to priors when unseen."""
        team = self.teams.set_index("constructor_id")["team_effect"]
        drv = self.drivers.set_index("driver_id")["shrunk_effect"]
        t = float(team.get(constructor_id, 0.0))
        d = float(drv.get(driver_id, 0.0))
        return t + d


def recency_weights(order_index: pd.Series, latest_index: float, half_life: float) -> pd.Series:
    """Exponential decay weights: an observation ``half_life`` races old counts 0.5."""
    age = (float(latest_index) - pd.to_numeric(order_index, errors="coerce")).clip(lower=0)
    return np.power(0.5, age / max(float(half_life), EPS))


def build_observations(
    results: pd.DataFrame,
    qualifying: pd.DataFrame,
    *,
    race_score: str = "rank",
) -> pd.DataFrame:
    """Assemble the long observation table the strength model is fitted on.

    Returns one row per (race, driver, observation-kind) with columns
    ``race_id, season, round, driver_id, constructor_id, kind, score, regime``.
    """
    # Retirements are *excluded* rather than scored as last: a mechanical failure
    # carries no information about pace, and scoring it as last would corrupt the
    # strength estimate of fast cars that happen to be unreliable. Reliability is
    # modelled separately in :mod:`f1pred.features.reliability`.
    if race_score not in {"rank", "pace"}:
        raise ValueError(f"unknown race_score {race_score!r}")
    if race_score == "pace":
        # Time, not order: a twenty-second win and a four-tenth win are different
        # evidence about the car, and the rank transform cannot tell them apart.
        from .racepace import race_pace_score

        race_obs = race_pace_score(results)
        race_obs = race_obs.loc[race_obs["score"].notna()].copy()
    else:
        race_obs = results.loc[results["classified"] == 1].copy()
        n_classified = race_obs.groupby("race_id")["driver_id"].transform("size")
        race_obs["score"] = normal_scores(race_obs["position_order"], n_classified)
    race_obs["kind"] = "race"

    qual_obs = qualifying.merge(
        results[["race_id", "driver_id", "season", "round"]].drop_duplicates(),
        on=["race_id", "driver_id", "season", "round"],
        how="left",
    )
    if "circuit_id" not in qual_obs:
        qual_obs = qual_obs.merge(
            results[["race_id", "circuit_id"]].drop_duplicates(), on="race_id", how="left"
        )
    qual_obs = qual_obs.loc[qual_obs["qual_gap_pct"].notna()].copy()
    qual_obs["score"] = qualifying_pace_score(qual_obs["qual_gap_pct"])
    qual_obs["kind"] = "qualifying"

    # ``circuit_id`` is carried through so that similarity-weighted (track-car
    # compatibility) fitting can re-weight observations by venue. Without it the
    # weighting silently degrades to a no-op.
    cols = [
        "race_id",
        "season",
        "round",
        "circuit_id",
        "driver_id",
        "constructor_id",
        "kind",
        "score",
        "event_index",
    ]
    for frame in (race_obs, qual_obs):
        if "circuit_id" not in frame:
            frame["circuit_id"] = pd.NA
    obs = pd.concat([race_obs[cols], qual_obs[cols]], ignore_index=True)
    obs["regime"] = obs["season"].map(regime_for)
    if obs["event_index"].isna().any():
        raise ValueError("observations must carry the panel's event_index")
    if obs.groupby("race_id")["event_index"].nunique().gt(1).any():
        raise ValueError("conflicting event_index values for one race")
    return obs


def fit_strength(
    observations: pd.DataFrame,
    *,
    as_of_index: float,
    target_regime: str,
    config: ModelConfig = DEFAULT_MODEL_CONFIG,
    kind_weights: dict[str, float] | None = None,
    observation_weights: pd.Series | None = None,
) -> StrengthEstimate:
    """Weighted ridge decomposition of observed scores into team + driver effects.

    The design matrix holds constructor and driver indicators. Equal ridge
    penalties choose a decomposition even without driver transfers; they do not
    identify separate causal effects. Use the experimental state-space model
    for explicit within-lineup driver contrasts.

    Parameters
    ----------
    as_of_index:
        Chronological index of the race being predicted. Only observations with a
        strictly smaller index are used, which is the leakage guarantee.
    target_regime:
        Observations from a different regime are multiplied by
        ``config.cross_regime_weight``.
    kind_weights:
        Relative trust in each observation kind. Qualifying is the cleaner pace
        measurement; race results carry race-pace and raceability information.
    observation_weights:
        Optional extra multiplier per observation, aligned on the observation
        index. Used by :mod:`f1pred.features.compatibility` to up-weight circuits
        similar to the target, which is how track-car compatibility enters the
        model without inventing an unidentifiable hand-built car vector.
    """
    kind_weights = kind_weights or {"qualifying": 1.0, "race": 0.8}
    obs = observations.loc[observations["event_index"] < as_of_index].copy()
    obs = obs.loc[obs["score"].notna()]
    if obs.empty:
        return StrengthEstimate(
            drivers=pd.DataFrame(
                columns=["driver_id", "driver_effect", "n_obs", "shrunk_effect", "effect_se"]
            ),
            teams=pd.DataFrame(columns=["constructor_id", "team_effect", "n_obs", "effect_se"]),
            as_of=str(as_of_index),
            n_observations=0,
            intercept=0.0,
        )

    weights = recency_weights(obs["event_index"], as_of_index, config.recency_half_life_races)
    weights = weights * obs["kind"].map(kind_weights).fillna(0.5)
    weights = weights * np.where(obs["regime"].eq(target_regime), 1.0, config.cross_regime_weight)
    if observation_weights is not None:
        weights = weights * observation_weights.reindex(obs.index).fillna(1.0).astype(float)
    obs["weight"] = weights.astype(float)
    obs = obs.loc[obs["weight"] > 1e-6]
    if obs.empty or obs["driver_id"].nunique() < 2:
        return StrengthEstimate(
            drivers=pd.DataFrame(
                columns=["driver_id", "driver_effect", "n_obs", "shrunk_effect", "effect_se"]
            ),
            teams=pd.DataFrame(columns=["constructor_id", "team_effect", "n_obs", "effect_se"]),
            as_of=str(as_of_index),
            n_observations=len(obs),
            intercept=0.0,
        )

    teams = sorted(obs["constructor_id"].unique())
    drivers = sorted(obs["driver_id"].unique())
    team_idx = {t: i for i, t in enumerate(teams)}
    driver_idx = {d: i for i, d in enumerate(drivers)}

    n, p_t, p_d = len(obs), len(teams), len(drivers)
    design = np.zeros((n, p_t + p_d + 1), dtype=float)
    design[:, 0] = 1.0
    design[np.arange(n), 1 + obs["constructor_id"].map(team_idx).to_numpy()] = 1.0
    design[np.arange(n), 1 + p_t + obs["driver_id"].map(driver_idx).to_numpy()] = 1.0

    y = obs["score"].to_numpy(dtype=float)
    w = obs["weight"].to_numpy(dtype=float)

    # Weighted ridge with an un-penalised intercept. Sum-to-zero is achieved by
    # penalising effects toward 0, which is the identifying constraint here.
    penalty = np.full(design.shape[1], config.ridge_alpha)
    penalty[0] = 0.0
    xtw = design.T * w
    gram = xtw @ design + np.diag(penalty)
    coef = np.linalg.solve(gram, xtw @ y)

    intercept = float(coef[0])
    # Ridge posterior covariance, sigma^2 (X'WX + lambda I)^-1. This is conditional
    # on treating recency weights as observation precision and the chosen ridge
    # penalty as a prior. It excludes hyperparameter uncertainty and precedes
    # the additional driver shrinkage below; it is not the covariance of that
    # final shrunk predictor. Do not draw these marginals independently for ranks.
    residual = y - design @ coef
    dof = max(w.sum() - np.trace(np.linalg.solve(gram, xtw @ design)), 1.0)
    residual_sd = float(np.sqrt(max((w * residual**2).sum() / dof, EPS)))
    covariance = residual_sd**2 * np.linalg.inv(gram)
    effect_se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    team_effects = pd.DataFrame(
        {
            "constructor_id": teams,
            "team_effect": coef[1 : 1 + p_t],
            "n_obs": [float(obs.loc[obs["constructor_id"] == t, "weight"].sum()) for t in teams],
            "effect_se": effect_se[1 : 1 + p_t],
        }
    )
    driver_effects = pd.DataFrame(
        {
            "driver_id": drivers,
            "driver_effect": coef[1 + p_t :],
            "n_obs": [float(obs.loc[obs["driver_id"] == d, "weight"].sum()) for d in drivers],
            "effect_se": effect_se[1 + p_t :],
        }
    )
    # Additional hand-set shrinkage toward zero for sparse drivers (not fitted EB).
    k = config.driver_shrinkage_races
    driver_effects["shrunk_effect"] = driver_effects["driver_effect"] * (
        driver_effects["n_obs"] / (driver_effects["n_obs"] + k)
    )
    return StrengthEstimate(
        drivers=driver_effects,
        teams=team_effects,
        as_of=str(as_of_index),
        n_observations=n,
        intercept=intercept,
        residual_sd=residual_sd,
        covariance=covariance,
        team_rows={t: 1 + i for i, t in enumerate(teams)},
        driver_rows={d: 1 + p_t + i for i, d in enumerate(drivers)},
    )


def entry_strength(
    estimate: StrengthEstimate,
    entries: pd.DataFrame,
    *,
    rookie_penalty: float = -0.25,
) -> pd.DataFrame:
    """Score a set of (driver, constructor) entries with the fitted strengths.

    Unknown drivers receive the team effect plus ``rookie_penalty`` and are
    flagged so the simulator can widen their uncertainty.
    """
    team_map = estimate.teams.set_index("constructor_id")["team_effect"]
    drv_map = estimate.drivers.set_index("driver_id")["shrunk_effect"]
    obs_map = estimate.drivers.set_index("driver_id")["n_obs"]

    out = entries.copy()
    out["team_effect"] = out["constructor_id"].map(team_map)
    out["driver_effect"] = out["driver_id"].map(drv_map)
    out["driver_obs_weight"] = out["driver_id"].map(obs_map).fillna(0.0)
    out["is_rookie"] = out["driver_effect"].isna()
    # An unseen team is worse-identified than an unseen driver: use the field mean.
    out["team_effect"] = out["team_effect"].fillna(0.0)
    out["driver_effect"] = out["driver_effect"].fillna(rookie_penalty)
    out["strength"] = out["team_effect"] + out["driver_effect"]

    # Uncertainty in the *estimate*, from the ridge posterior rather than a chosen
    # formula. The variance of the sum, not the sum of variances: team and driver
    # effects are confounded and strongly negatively correlated, so adding them in
    # quadrature would roughly double the width that the data actually supports.
    if estimate.covariance is not None:
        out["strength_se"] = [
            float(np.sqrt(max(estimate.strength_variance(d, c), 0.0)))
            for d, c in zip(out["driver_id"], out["constructor_id"], strict=True)
        ]
    else:
        out["strength_se"] = np.nan
    return out


def fit_strength_backfit(
    observations: pd.DataFrame,
    *,
    as_of_index: float,
    target_regime: str,
    config: ModelConfig = DEFAULT_MODEL_CONFIG,
    kind_weights: dict[str, float] | None = None,
    team_half_life: float | None = None,
    driver_half_life: float | None = None,
    n_passes: int = 4,
) -> StrengthEstimate:
    """Additive decomposition with **separate recency half-lives** per component.

    A constructor's competitiveness moves on the timescale of an upgrade cycle -
    a few races. A driver's underlying skill moves on the timescale of seasons.
    Forcing both to share one half-life is a modelling compromise that costs
    reactivity: the joint fit kept rating a team on pace it no longer had.

    One weighted ridge cannot carry two different weight vectors, so the additive
    model is fitted by **backfitting**: team effects are estimated on the
    driver-adjusted residual using short-memory weights, driver effects on the
    team-adjusted residual using long-memory weights, alternating until stable.

    Parameters
    ----------
    team_half_life, driver_half_life:
        Half-lives in races. Default to ``config.recency_half_life_races`` so the
        function reduces to the joint fit when they are equal.
    n_passes:
        Backfitting iterations. Four is ample; the effects move by <1e-3 after the
        third on real data.
    """
    kind_weights = kind_weights or {"qualifying": 1.0, "race": 0.8}
    team_hl = float(team_half_life or config.recency_half_life_races)
    driver_hl = float(driver_half_life or config.recency_half_life_races)

    obs = observations.loc[observations["event_index"] < as_of_index].copy()
    obs = obs.loc[obs["score"].notna()]
    if obs.empty or obs["driver_id"].nunique() < 2:
        return _empty_estimate(as_of_index)

    kind_w = obs["kind"].map(kind_weights).fillna(0.5).to_numpy(float)
    regime_w = np.where(obs["regime"].eq(target_regime), 1.0, config.cross_regime_weight)
    w_team = (
        recency_weights(obs["event_index"], as_of_index, team_hl).to_numpy(float)
        * kind_w
        * regime_w
    )
    w_driver = (
        recency_weights(obs["event_index"], as_of_index, driver_hl).to_numpy(float)
        * kind_w
        * regime_w
    )
    keep = (w_team > 1e-6) | (w_driver > 1e-6)
    obs, w_team, w_driver = obs.loc[keep], w_team[keep], w_driver[keep]
    if obs.empty:
        return _empty_estimate(as_of_index)

    y = obs["score"].to_numpy(float)
    teams = sorted(obs["constructor_id"].unique())
    drivers = sorted(obs["driver_id"].unique())
    t_code = obs["constructor_id"].map({t: i for i, t in enumerate(teams)}).to_numpy()
    d_code = obs["driver_id"].map({d: i for i, d in enumerate(drivers)}).to_numpy()

    def weighted_ridge_means(
        codes: np.ndarray, target: np.ndarray, weights: np.ndarray, n_levels: int
    ) -> np.ndarray:
        """Per-level weighted mean shrunk toward 0 by the ridge penalty."""
        num = np.bincount(codes, weights=weights * target, minlength=n_levels)
        den = np.bincount(codes, weights=weights, minlength=n_levels) + config.ridge_alpha
        return num / np.maximum(den, EPS)

    intercept = float(np.average(y, weights=np.maximum(w_team, EPS)))
    team_eff = np.zeros(len(teams))
    driver_eff = np.zeros(len(drivers))
    for _ in range(int(n_passes)):
        team_eff = weighted_ridge_means(
            t_code, y - intercept - driver_eff[d_code], w_team, len(teams)
        )
        driver_eff = weighted_ridge_means(
            d_code, y - intercept - team_eff[t_code], w_driver, len(drivers)
        )
        # Re-centre so the intercept absorbs the overall level and effects stay identified.
        shift = float(
            np.average(
                driver_eff,
                weights=np.bincount(d_code, weights=w_driver, minlength=len(drivers)) + EPS,
            )
        )
        driver_eff = driver_eff - shift
        intercept = intercept + shift

    team_obs = np.bincount(t_code, weights=w_team, minlength=len(teams))
    driver_obs = np.bincount(d_code, weights=w_driver, minlength=len(drivers))
    team_effects = pd.DataFrame(
        {"constructor_id": teams, "team_effect": team_eff, "n_obs": team_obs}
    )
    driver_effects = pd.DataFrame(
        {"driver_id": drivers, "driver_effect": driver_eff, "n_obs": driver_obs}
    )
    k = config.driver_shrinkage_races
    driver_effects["shrunk_effect"] = driver_effects["driver_effect"] * (
        driver_effects["n_obs"] / (driver_effects["n_obs"] + k)
    )
    return StrengthEstimate(
        drivers=driver_effects,
        teams=team_effects,
        as_of=str(as_of_index),
        n_observations=len(obs),
        intercept=intercept,
    )


def _empty_estimate(as_of_index: float) -> StrengthEstimate:
    """Neutral estimate used when no legal observation exists yet."""
    return StrengthEstimate(
        drivers=pd.DataFrame(
            columns=["driver_id", "driver_effect", "n_obs", "shrunk_effect", "effect_se"]
        ),
        teams=pd.DataFrame(columns=["constructor_id", "team_effect", "n_obs", "effect_se"]),
        as_of=str(as_of_index),
        n_observations=0,
        intercept=0.0,
    )


@dataclass(frozen=True)
class DualStrength:
    """Separate one-lap and race-pace strengths for the same entries.

    Motivated by Kelly & co.'s RAPM decomposition of 2014-2024 Formula 1, which
    finds constructors explain ~64% of race-outcome variance but that the
    constructor share is *lower* in qualifying than in race trim. A single pooled
    strength therefore averages two genuinely different quantities. Splitting them
    also matches how the simulator actually uses them: the grid is decided by
    one-lap pace, the race by race pace.
    """

    qualifying: StrengthEstimate
    race: StrengthEstimate

    @property
    def n_observations(self) -> int:
        return self.qualifying.n_observations + self.race.n_observations


def fit_dual_strength(
    observations: pd.DataFrame,
    *,
    as_of_index: float,
    target_regime: str,
    config: ModelConfig = DEFAULT_MODEL_CONFIG,
) -> DualStrength:
    """Fit one-lap and race-pace strengths as two independent decompositions.

    Each half uses only its own observation kind, so the qualifying fit is a pure
    single-lap measurement (no strategy, traffic or attrition) and the race fit is
    a pure race-trim measurement.
    """
    qual_obs = observations.loc[observations["kind"] == "qualifying"]
    race_obs = observations.loc[observations["kind"] == "race"]
    return DualStrength(
        qualifying=fit_strength(
            qual_obs,
            as_of_index=as_of_index,
            target_regime=target_regime,
            config=config,
            kind_weights={"qualifying": 1.0},
        ),
        race=fit_strength(
            race_obs,
            as_of_index=as_of_index,
            target_regime=target_regime,
            config=config,
            kind_weights={"race": 1.0},
        ),
    )


def dual_entry_strength(
    dual: DualStrength,
    entries: pd.DataFrame,
    *,
    rookie_penalty: float = -0.25,
) -> pd.DataFrame:
    """Score entries with both strengths, returning ``qual_strength``/``race_strength``.

    ``strength`` is kept as the mean of the two so that downstream code with no
    interest in the split keeps working unchanged.
    """
    qual = entry_strength(dual.qualifying, entries, rookie_penalty=rookie_penalty)
    race = entry_strength(dual.race, entries, rookie_penalty=rookie_penalty)
    out = entries.copy()
    out["qual_strength"] = qual["strength"].to_numpy()
    out["race_strength"] = race["strength"].to_numpy()
    out["qual_driver_effect"] = qual["driver_effect"].to_numpy()
    out["race_driver_effect"] = race["driver_effect"].to_numpy()
    out["team_effect"] = race["team_effect"].to_numpy()
    out["is_rookie"] = (qual["is_rookie"] | race["is_rookie"]).to_numpy()
    out["driver_obs_weight"] = np.minimum(
        qual["driver_obs_weight"].to_numpy(), race["driver_obs_weight"].to_numpy()
    )
    out["strength"] = 0.5 * (out["qual_strength"] + out["race_strength"])
    # The two fits shrink by different amounts because the race fit sees noisier,
    # variable-size fields, so their raw spreads are not comparable. Re-scale both
    # to a common within-race spread: only the *ordering and relative gaps* inside
    # a race are meaningful to the simulator, and this removes a pure artefact of
    # ridge shrinkage rather than any real signal.
    for col in ("qual_strength", "race_strength"):
        values = out[col].astype(float)
        spread = values.std(ddof=0)
        out[f"{col}_z"] = (values - values.mean()) / (spread if spread > EPS else 1.0)
    return out
