from __future__ import annotations

import pandas as pd

STAGE_ORDER = {
    "PRE_WEEKEND": 0,
    "POST_FP1": 1,
    "POST_FP2": 2,
    "POST_FP3": 3,
    "POST_QUALI": 4,
    "RACE_START": 5,
    "POST_RACE": 6,
}

FEATURE_STAGE = {
    "historical_form": "PRE_WEEKEND",
    "track_geometry": "PRE_WEEKEND",
    "track_similarity": "PRE_WEEKEND",
    "archived_weather": "PRE_WEEKEND",
    "fp1": "POST_FP1",
    "fp2": "POST_FP2",
    "fp3": "POST_FP3",
    "qualifying": "POST_QUALI",
    "grid": "POST_QUALI",
    "race_weather": "POST_RACE",
    "race_stints": "POST_RACE",
    "pit_stops": "POST_RACE",
    "race_laps": "POST_RACE",
    "result": "POST_RACE",
}


def allowed(feature_family: str, prediction_stage: str) -> bool:
    need = FEATURE_STAGE.get(feature_family, "POST_RACE")
    return STAGE_ORDER[need] <= STAGE_ORDER[prediction_stage]


def assert_no_future_features(feature_families, prediction_stage):
    bad = [f for f in feature_families if not allowed(f, prediction_stage)]
    if bad:
        raise ValueError(f"Leakage risk at {prediction_stage}: {bad}")
    return True


def assert_timestamp_cutoff(df: pd.DataFrame, feature_time_col: str, prediction_time_col: str):
    f = pd.to_datetime(df[feature_time_col], utc=True, errors="coerce")
    p = pd.to_datetime(df[prediction_time_col], utc=True, errors="coerce")
    bad = (f > p) & f.notna() & p.notna()
    if bad.any():
        raise ValueError(f"{int(bad.sum())} feature rows occur after prediction timestamp")
    return True
