"""Shared fixtures. Synthetic data only, so the suite runs offline and fast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_results() -> pd.DataFrame:
    """Four seasons, 10 races each, 6 drivers on 3 teams with known true strength.

    Team A is fastest, team C slowest, and within each team the first driver is
    stronger. Recovering this ordering is the model's basic identification test.
    """
    rng = np.random.default_rng(7)
    true_team = {"team_a": 1.2, "team_b": 0.0, "team_c": -1.2}
    true_driver = {"d1": 0.4, "d2": -0.4, "d3": 0.3, "d4": -0.3, "d5": 0.2, "d6": -0.2}
    lineup = {
        "d1": "team_a",
        "d2": "team_a",
        "d3": "team_b",
        "d4": "team_b",
        "d5": "team_c",
        "d6": "team_c",
    }

    rows = []
    for season in (2022, 2023, 2024, 2025):
        for rnd in range(1, 11):
            race_id = f"{season}_{rnd:02d}"
            latent = {
                d: true_team[t] + true_driver[d] + rng.normal(0, 0.35) for d, t in lineup.items()
            }
            order = sorted(latent, key=lambda d: -latent[d])
            for pos, d in enumerate(order, start=1):
                rows.append(
                    {
                        "race_id": race_id,
                        "season": season,
                        "round": rnd,
                        "circuit_id": f"circuit_{rnd}",
                        "date": f"{season}-{rnd:02d}-01",
                        "driver_id": d,
                        "driver_code": d.upper(),
                        "constructor_id": lineup[d],
                        "grid": pos,
                        "position_order": pos,
                        "position_text": str(pos),
                        "status": "Finished",
                        "classified": 1,
                        "started": 1,
                        "dnf": 0,
                        "points": max(0, 26 - pos * 4),
                        "laps": 60,
                        "winner": int(pos == 1),
                        "regime": "GROUND_EFFECT_2022" if season < 2026 else "NEW_PU_AERO_2026",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_qualifying(synthetic_results: pd.DataFrame) -> pd.DataFrame:
    """Qualifying consistent with the synthetic race order."""
    q = synthetic_results[
        ["race_id", "season", "round", "driver_id", "constructor_id", "grid"]
    ].copy()
    q = q.rename(columns={"grid": "qual_position"})
    q["qual_best_s"] = 80.0 + 0.25 * (q["qual_position"] - 1)
    pole = q.groupby("race_id")["qual_best_s"].transform("min")
    q["qual_gap_pct"] = 100.0 * (q["qual_best_s"] - pole) / pole
    return q


@pytest.fixture
def synthetic_schedule(synthetic_results: pd.DataFrame) -> pd.DataFrame:
    """Calendar matching the synthetic results."""
    s = synthetic_results[["season", "round", "race_id", "date", "circuit_id"]].drop_duplicates()
    return s.sort_values(["season", "round"]).reset_index(drop=True)
