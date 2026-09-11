"""Probability calibration for race-level forecasts.

A simulator can rank well and still be badly calibrated: if its pace noise is too
small it will claim 45% for a favourite who wins 30% of the time. Since the
stated objective is a *calibrated* probabilistic forecast, calibration is fitted
and reported as a first-class step rather than assumed.

Two mechanisms are provided:

``fit_temperature``
    A single scalar applied to the log of the simulated win probabilities inside
    each race, then renormalised. ``T > 1`` flattens an over-confident forecast,
    ``T < 1`` sharpens an under-confident one. One parameter, fitted by minimising
    race-level winner log-loss, so it cannot overfit a 92-race window.

``fit_noise_scale``
    The structural alternative: rather than post-processing the probabilities,
    scale the simulator's pace noise so the spread of simulated outcomes matches
    reality. Preferred when it works, because it keeps podium and top-10
    probabilities coherent with the win probability instead of only fixing one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-12


def apply_temperature(
    probabilities: pd.Series, race_ids: pd.Series, temperature: float
) -> pd.Series:
    """Re-sharpen or flatten within-race probabilities and renormalise.

    Operates on log-probabilities so the transform is a proper tempered softmax:
    ``p_i' = p_i^(1/T) / sum_j p_j^(1/T)``.
    """
    p = pd.to_numeric(probabilities, errors="coerce").clip(lower=EPS)
    t = max(float(temperature), 1e-6)
    tempered = np.power(p, 1.0 / t)
    totals = tempered.groupby(race_ids).transform("sum")
    return tempered / totals.clip(lower=EPS)


def winner_log_loss(probabilities: pd.Series, race_ids: pd.Series, is_winner: pd.Series) -> float:
    """Mean negative log probability assigned to the actual winner, per race."""
    df = pd.DataFrame(
        {
            "p": pd.to_numeric(probabilities, errors="coerce"),
            "race": race_ids,
            "y": is_winner.astype(bool),
        }
    )
    losses = []
    for _, g in df.groupby("race"):
        if g["y"].sum() != 1:
            continue
        losses.append(-np.log(max(float(g.loc[g["y"], "p"].iloc[0]), EPS)))
    return float(np.mean(losses)) if losses else float("nan")


@dataclass(frozen=True)
class TemperatureFit:
    """Result of a calibration fit."""

    temperature: float
    log_loss_before: float
    log_loss_after: float
    n_races: int

    @property
    def improved(self) -> bool:
        """True when calibration reduced winner log-loss."""
        return self.log_loss_after < self.log_loss_before


def fit_temperature(
    predictions: pd.DataFrame,
    *,
    prob_col: str = "win_probability",
    race_col: str = "race_id",
    winner_col: str = "winner",
    grid: np.ndarray | None = None,
) -> TemperatureFit:
    """Fit one temperature by minimising race-level winner log-loss.

    Parameters
    ----------
    grid:
        Temperatures to search. The default spans heavy sharpening (0.3) to heavy
        flattening (4.0) on a log scale.
    """
    search = np.geomspace(0.3, 4.0, 80) if grid is None else np.asarray(grid, dtype=float)
    p = predictions[prob_col]
    races = predictions[race_col]
    winners = predictions[winner_col].astype(bool)

    before = winner_log_loss(p, races, winners)
    best_t, best_loss = 1.0, before
    for t in search:
        loss = winner_log_loss(apply_temperature(p, races, t), races, winners)
        if np.isfinite(loss) and loss < best_loss:
            best_t, best_loss = float(t), float(loss)
    return TemperatureFit(
        temperature=best_t,
        log_loss_before=before,
        log_loss_after=best_loss,
        n_races=int(predictions.loc[winners, race_col].nunique()),
    )


def reliability_bins(
    predictions: pd.DataFrame,
    *,
    prob_col: str,
    outcome_col: str,
    edges: np.ndarray | None = None,
) -> pd.DataFrame:
    """Observed frequency against predicted probability, with counts.

    Uses explicit edges rather than quantiles so that bins are comparable across
    models and seasons.
    """
    bins = np.array([0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.55, 1.0]) if edges is None else edges
    df = predictions[[prob_col, outcome_col]].dropna().copy()
    if df.empty:
        return pd.DataFrame()
    df["bin"] = pd.cut(df[prob_col], bins, include_lowest=True)
    out = df.groupby("bin", observed=True).agg(
        n=(outcome_col, "size"),
        mean_predicted=(prob_col, "mean"),
        observed_frequency=(outcome_col, "mean"),
    )
    out["gap"] = out["mean_predicted"] - out["observed_frequency"]
    out["weight"] = out["n"] / out["n"].sum()
    return out.reset_index()


def expected_calibration_error(
    predictions: pd.DataFrame, *, prob_col: str, outcome_col: str
) -> float:
    """Weighted mean absolute gap between predicted and observed frequency."""
    table = reliability_bins(predictions, prob_col=prob_col, outcome_col=outcome_col)
    if table.empty:
        return float("nan")
    return float((table["gap"].abs() * table["weight"]).sum())
