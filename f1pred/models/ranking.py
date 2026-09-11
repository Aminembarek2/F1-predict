"""Chronological probability calibration and a Plackett–Luce benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax

from ..config import SEED
from ..sim.race import conditional_dnf_rates


@dataclass(frozen=True)
class RankCalibration:
    temperature: float
    n_races: int
    last_training_index: float | None
    objective: str


def fit_rank_calibration(
    history: pd.DataFrame,
    *,
    as_of_index: float,
    score_col: str = "score",
    objective: str = "winner",
) -> RankCalibration:
    """Fit a scale using strictly earlier out-of-fold race predictions.

    Winner calibration minimizes race log-loss; the Plackett–Luce variant fits
    the likelihood of each classified finishing order, averaged per race.
    """
    if objective not in {"winner", "ranking"}:
        raise ValueError("objective must be winner or ranking")
    past = history.loc[history["event_index"] < as_of_index].copy()
    groups = []
    for _, group in past.groupby("race_id"):
        if group["winner"].sum() != 1 or not np.isfinite(group[score_col]).all():
            raise ValueError("calibration requires complete scores and one winner per race")
        if objective == "ranking":
            group = group.loc[group["classified"].eq(1)].sort_values("position_order")
        if len(group) >= 2:
            groups.append((group[score_col].to_numpy(float), group["winner"].to_numpy(bool)))
    last = float(past["event_index"].max()) if len(past) else None
    if len(groups) < 10:
        return RankCalibration(1.0, len(groups), last, objective)

    def loss(log_temperature: float) -> float:
        temperature = np.exp(log_temperature)
        losses = []
        for score, winner in groups:
            z = score / temperature
            if objective == "winner":
                losses.append(logsumexp(z) - float(z[winner][0]))
            else:
                # Each suffix is the set of drivers still available for selection.
                normalizers = np.logaddexp.accumulate(z[::-1])[::-1]
                losses.append(float(np.mean(normalizers[:-1] - z[:-1])))
        return float(np.mean(losses))

    fitted = minimize_scalar(loss, bounds=(np.log(0.05), np.log(20)), method="bounded")
    if not fitted.success:
        raise RuntimeError("probability calibration did not converge")
    return RankCalibration(float(np.exp(fitted.x)), len(groups), last, objective)


def rank_probabilities(scores: pd.Series, temperature: float) -> np.ndarray:
    """Normalized winner probabilities; no independent per-driver classifiers."""
    values = scores.to_numpy(float)
    if not np.isfinite(values).all() or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("scores and temperature must be finite, with temperature positive")
    return softmax(values / temperature)


def simulate_plackett_luce(
    entries: pd.DataFrame, *, temperature: float, n_sims: int = 10000, seed: int = SEED
) -> pd.DataFrame:
    """Draw full Plackett–Luce orders using Gumbel noise, plus marginal DNF risk.

    Uses the same correlated-retirement contract as the incumbent so that the
    ranking distribution is the principal difference between these models.
    """
    if n_sims <= 0 or temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("positive simulation count and temperature required")
    frame = entries.sort_values("driver_id").reset_index(drop=True).copy()
    n = len(frame)
    if not n:
        raise ValueError("no entries")
    rng = np.random.default_rng(seed)
    score = frame["strength"].to_numpy(float) / temperature
    calm, chaos = conditional_dnf_rates(frame["dnf_probability"].to_numpy(float), 0.22, 2.1)
    performance = score + rng.gumbel(size=(n_sims, n))
    hazard = np.where(rng.random((n_sims, 1)) < 0.22, chaos, calm)
    retired = rng.random((n_sims, n)) < hazard
    performance = np.where(retired, performance - 1e6, performance)
    positions = np.argsort(np.argsort(-performance, axis=1), axis=1) + 1
    frame["win_probability"] = (positions == 1).mean(axis=0)
    frame["podium_probability"] = (positions <= 3).mean(axis=0)
    frame["top10_probability"] = (positions <= min(n, 10)).mean(axis=0)
    frame["dnf_probability_sim"] = retired.mean(axis=0)
    frame["expected_finish"] = positions.mean(axis=0)
    frame["score"] = -np.where(retired, 0, positions).sum(axis=0) / np.maximum(
        (~retired).sum(axis=0), 1
    )
    frame["pace_rank_score"] = frame["score"]
    frame["expected_finish_if_classified"] = -frame["score"]
    frame["probability_scope"] = "full_ranking"
    for quantile in (10, 50, 90):
        frame[f"finish_p{quantile}"] = np.quantile(
            positions, quantile / 100, axis=0, method="inverted_cdf"
        )
    frame = frame.sort_values(["score", "driver_id"], ascending=[False, True]).reset_index(
        drop=True
    )
    frame["predicted_rank"] = np.arange(1, n + 1)
    frame["win_probability_rank"] = (
        frame["win_probability"].rank(ascending=False, method="min").astype(int)
    )
    return frame
