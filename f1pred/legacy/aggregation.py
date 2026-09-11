from __future__ import annotations

import numpy as np
import pandas as pd

from .features import cosine_compatibility, track_demand_vector


def aggregate_driver_corners(corners: pd.DataFrame) -> pd.DataFrame:
    c = corners.copy()
    c["speed_class"] = pd.cut(
        pd.to_numeric(c.apex_speed_kmh, errors="coerce"),
        [-np.inf, 140, 220, np.inf],
        labels=["low", "medium", "high"],
    )
    rows = []
    for driver, g in c.groupby("driver"):
        row = {
            "driver": driver,
            "braking_strength": pd.to_numeric(g.braking_efficiency_ms2, errors="coerce").mean(),
            "traction_strength": pd.to_numeric(g.traction_index_ms2, errors="coerce").mean(),
            "aero_commitment": pd.to_numeric(g.aero_commitment, errors="coerce").mean(),
            "exit_amplification": pd.to_numeric(g.exit_amplification, errors="coerce").mean(),
        }
        for cls in ["low", "medium", "high"]:
            z = g[g.speed_class == cls]
            row[f"{cls}_speed_performance"] = (
                pd.to_numeric(z.corner_performance, errors="coerce").mean() if len(z) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_track_vector(corners: pd.DataFrame) -> pd.Series:
    # Deduplicate repeated driver observations per corner before constructing circuit demand.
    cols = [
        "corner",
        "apex_speed_kmh",
        "braking_efficiency_ms2",
        "traction_index_ms2",
        "aero_commitment",
        "exit_amplification",
    ]
    t = corners[cols].groupby("corner", as_index=False).median(numeric_only=True)
    return track_demand_vector(t)


def compatibility_from_named_vectors(track: pd.Series, car: pd.Series) -> float:
    # Match semantic dimensions rather than relying on arbitrary column order.
    pairs = [
        ("low_speed_share", "low_speed"),
        ("medium_speed_share", "medium_speed"),
        ("high_speed_share", "high_speed"),
        ("braking_demand", "braking"),
        ("traction_demand", "traction"),
        ("aero_demand", "aero"),
    ]
    a = []
    b = []
    for tk, ck in pairs:
        if tk in track and ck in car:
            a.append(track[tk])
            b.append(car[ck])
    return cosine_compatibility(a, b)
