"""Data-source adapters, and the contract they all have to meet.

Two sources supply practice laps and they disagree about types: one returns
timestamps as datetimes, the other as ISO strings. Left alone that difference
survives all the way to the parquet writer, which then refuses a mixed column —
so the contract is enforced here, at the boundary, rather than trusted.
"""

from __future__ import annotations

import pandas as pd

#: Timestamp columns every practice-lap frame carries, in UTC.
PRACTICE_TIMESTAMPS: tuple[str, ...] = (
    "date_start",
    "observed_at",
    "session_start",
    "cutoff_utc",
)

#: Columns the practice feature and the experiment runner actually read. A source
#: may carry extras; it may not omit one of these.
PRACTICE_REQUIRED: tuple[str, ...] = (
    "race_id",
    "season",
    "stage",
    "driver_id",
    "lap_duration",
    "tyre_age",
    "compound",
    "stint_number",
    "is_pit_out_lap",
    "interrupted",
    "rainfall",
    *PRACTICE_TIMESTAMPS,
)


def normalise_practice(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce one source's practice laps into the shared contract.

    Timestamps become UTC datetimes and the numeric and boolean columns become
    their declared types, so frames from different sources concatenate without
    silently producing an object column.
    """
    if frame.empty:
        return frame
    missing = set(PRACTICE_REQUIRED) - set(frame)
    if missing:
        raise ValueError(f"practice source is missing {sorted(missing)}")
    out = frame.copy()
    for column in PRACTICE_TIMESTAMPS:
        out[column] = pd.to_datetime(out[column], utc=True, format="mixed")
    for column in ("lap_duration", "tyre_age", "stint_number", "rainfall"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ("is_pit_out_lap", "interrupted"):
        out[column] = out[column].astype(bool)
    out["compound"] = out["compound"].astype(str).str.upper()
    return out.reset_index(drop=True)
