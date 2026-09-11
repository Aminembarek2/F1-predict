"""Track-car compatibility through similarity-weighted history.

The question this module answers is the project's central one: *which car suits
this circuit*, rather than *which car has been fastest on average*. A team that is
strong in high-downforce corner sequences and weak on straights should be rated
differently for Budapest than for Monza.

Method
------
Rather than asserting a hand-built ``TRACK_VECTOR`` and ``CAR_VECTOR`` and taking
their cosine - which is unidentifiable from public data and, on all-positive
vectors, nearly constant - compatibility is obtained by **re-weighting history**:
observations gathered at circuits similar to the target count for more.

Similarity uses the standardised circuit profile (average lap speed, lap length,
qualifying field spread, overtaking difficulty, street/permanent), converted to a
weight by a Gaussian kernel. The bandwidth controls how sharply the model
specialises; a very small bandwidth reduces to circuit history (which the original
baseline showed does not work), and a very large one reduces to the pooled
model. The bandwidth is therefore a hyper-parameter to be tuned, not assumed.

Availability: ``PRE_WEEKEND`` - all inputs are historical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9

#: Profile columns used for similarity. All are derived from data, not copied in.
SIMILARITY_FEATURES = [
    "avg_lap_speed_kmh",
    "lap_length_m",
    "field_spread_pct",
    "overtaking_difficulty",
]


def circuit_similarity(
    target: pd.Series,
    profiles: pd.DataFrame,
    *,
    features: list[str] | None = None,
    bandwidth: float = 1.0,
    street_bonus: float = 0.35,
) -> pd.Series:
    """Gaussian-kernel similarity between one target circuit and every other.

    Parameters
    ----------
    target:
        Profile row for the target circuit; missing features are ignored.
    profiles:
        Circuit profile table indexed by ``circuit_id``.
    bandwidth:
        Kernel width in standardised units. Larger means less specialisation.
    street_bonus:
        Extra similarity granted to circuits of the same layout family, expressed
        as a reduction in squared distance. Street circuits share wall proximity,
        low grip and limited run-off regardless of their speed profile.

    Returns
    -------
    pd.Series
        Similarity in (0, 1] indexed by ``circuit_id``, with the target itself
        scoring 1.0 when present.
    """
    cols = [c for c in (features or SIMILARITY_FEATURES) if c in profiles and c in target.index]
    pool = profiles.dropna(subset=cols).copy()
    if pool.empty or not cols:
        return pd.Series(
            1.0, index=profiles["circuit_id"] if "circuit_id" in profiles else profiles.index
        )

    mu = pool[cols].mean()
    sd = pool[cols].std(ddof=0).replace(0, 1.0)
    z_pool = (pool[cols] - mu) / sd
    z_target = (target[cols].astype(float) - mu) / sd
    squared = ((z_pool - z_target) ** 2).sum(axis=1)

    if "is_street" in pool and "is_street" in target.index:
        same_family = pool["is_street"].astype(bool) == bool(target["is_street"])
        squared = squared - np.where(same_family, street_bonus, 0.0)
    squared = squared.clip(lower=0.0)

    weights = np.exp(-squared / (2.0 * max(bandwidth, EPS) ** 2))
    index = pool["circuit_id"] if "circuit_id" in pool else pool.index
    return pd.Series(weights.to_numpy(), index=index, name="similarity")


def similarity_weighted_observations(
    observations: pd.DataFrame,
    similarity: pd.Series,
    *,
    floor: float = 0.15,
    circuit_col: str = "circuit_id",
) -> pd.Series:
    """Per-observation multiplier from circuit similarity.

    A ``floor`` keeps dissimilar circuits contributing something: the pooled
    signal is strong and discarding it entirely reproduces the circuit-history
    failure mode of the original circuit-history baseline.
    """
    if circuit_col not in observations:
        return pd.Series(1.0, index=observations.index)
    mapped = observations[circuit_col].map(similarity)
    return mapped.fillna(similarity.median() if len(similarity) else 1.0).clip(lower=floor)


def build_target_profile(
    circuit_id: str,
    profiles: pd.DataFrame,
    *,
    published_lap_length_m: float | None = None,
    published_pole_time_s: float | None = None,
    is_street: bool | None = None,
    overtaking_difficulty: float | None = None,
) -> pd.Series:
    """Profile row for the target circuit, synthesising one when it has no history."""
    existing = (
        profiles.loc[profiles["circuit_id"] == circuit_id]
        if "circuit_id" in profiles
        else profiles.iloc[0:0]
    )
    if len(existing) and existing[SIMILARITY_FEATURES].notna().all(axis=1).iloc[0]:
        return existing.iloc[0]

    if not (published_lap_length_m and published_pole_time_s):
        raise ValueError(f"{circuit_id} has no history and no published specification")
    return pd.Series(
        {
            "circuit_id": circuit_id,
            "lap_length_m": float(published_lap_length_m),
            "avg_lap_speed_kmh": 3.6 * published_lap_length_m / published_pole_time_s,
            "field_spread_pct": float(profiles["field_spread_pct"].median()),
            "overtaking_difficulty": float(
                overtaking_difficulty
                if overtaking_difficulty is not None
                else profiles["overtaking_difficulty"].median()
            ),
            "is_street": bool(is_street) if is_street is not None else False,
        }
    )
