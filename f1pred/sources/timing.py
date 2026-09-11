"""Practice laps from the public F1 timing archive, via FastF1.

OpenF1 restricts even historical endpoints to paying accounts whenever a session
is live, so a second public source is needed for the practice experiment to be
reproducible on any day. This adapter emits exactly the column contract that
:func:`f1pred.sources.openf1.load_practice` emits, so the two are interchangeable
and the experiment never learns which one supplied the laps.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import CACHE_DIR
from . import normalise_practice

log = logging.getLogger(__name__)

# FP2 is the last long-run session of a conventional weekend and the only
# practice session a sprint weekend does not have; the experiment is declared on
# FP2 alone so that sprint and conventional weekends are never mixed.
SESSION = "FP2"
# Laps become usable once the session is over. The published FP2 length is one
# hour; the extra margin covers a red-flagged session running long.
AVAILABILITY_MARGIN = pd.to_timedelta(5, unit="m")


def _enable_cache() -> None:
    import fastf1

    cache = CACHE_DIR / "fastf1"
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))


def load_practice(
    seasons: list[int], schedule: pd.DataFrame, results: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch FP2 laps with stints, tyre age, weather and lap-validity flags.

    Weekends are matched on season and round, which both sources key on, rather
    than on a hand-maintained circuit mapping. A weekend whose session is absent
    or unmatched is reported in the coverage table and never replaced by
    synthetic laps.
    """
    try:
        import fastf1
        from fastf1.exceptions import (
            DataNotLoadedError,
            InvalidSessionError,
            NoLapDataError,
            RateLimitExceededError,
        )
    except ImportError as exc:  # pragma: no cover - exercised by the install extra
        raise RuntimeError("practice ingestion needs the 'telemetry' extra (fastf1)") from exc

    _enable_cache()
    frames: list[pd.DataFrame] = []
    coverage: list[dict] = []
    fetched_at = pd.Timestamp.now(tz="UTC").isoformat()
    for season in seasons:
        if season < 2018:
            coverage.append({"season": season, "status": "unavailable_before_2018"})
            continue
        races = schedule.loc[schedule["season"].eq(season)]
        codes = results.loc[results["season"].eq(season), ["driver_code", "driver_id"]]
        codes = codes.drop_duplicates()
        if codes["driver_code"].duplicated().any():
            raise ValueError(f"ambiguous driver abbreviation in {season}")
        for race in races.to_dict("records"):
            race_id = race["race_id"]
            try:
                session = fastf1.get_session(season, int(race["round"]), SESSION)
                session.load(telemetry=False, weather=True, messages=False, livedata=None)
                laps = session.laps
            except RateLimitExceededError:
                # The public mirror allows 500 requests an hour. Stop cleanly and
                # keep what was downloaded: cached sessions cost nothing on the
                # next run, so ingestion resumes where it left off.
                log.warning("rate limit reached at %s; returning partial coverage", race_id)
                coverage.append({"race_id": race_id, "status": "rate_limited"})
                return _finish(frames, coverage)
            except (
                ValueError,
                KeyError,
                DataNotLoadedError,
                InvalidSessionError,
                NoLapDataError,
            ) as exc:
                # 2023-2025 sprint weekends replaced FP2 with a second qualifying,
                # so a missing session here is the schedule, not a download failure.
                coverage.append(
                    {"race_id": race_id, "status": "no_fp2_session", "detail": type(exc).__name__}
                )
                continue
            if laps is None or laps.empty:
                coverage.append({"race_id": race_id, "status": "no_laps_published"})
                continue
            frame = _normalise(laps, session, codes)
            if frame.empty:
                coverage.append({"race_id": race_id, "status": "no_identifiable_laps"})
                continue
            frame["race_id"], frame["season"], frame["stage"] = race_id, season, "POST_FP2"
            frame["source"] = f"fastf1:{season}:{int(race['round'])}:{SESSION}"
            frame["fetched_at"] = fetched_at
            frames.append(frame)
            coverage.append({"race_id": race_id, "status": "downloaded", "laps": len(frame)})
            log.info("practice ingestion %s: %d laps", race_id, len(frame))
    return _finish(frames, coverage)


def _finish(frames: list[pd.DataFrame], coverage: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    laps = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return normalise_practice(laps), pd.DataFrame(coverage)


def _flag(column: pd.Series) -> np.ndarray:
    """Read an optional boolean timing flag, treating a missing value as False."""
    return column.astype("object").where(column.notna(), False).astype(bool).to_numpy()


def _normalise(laps: pd.DataFrame, session: object, codes: pd.DataFrame) -> pd.DataFrame:
    """Rewrite one session into the shared practice-lap contract."""
    start = pd.Timestamp(session.date).tz_localize("UTC")  # session start, published in UTC
    frame = pd.DataFrame(
        {
            "driver_code": laps["Driver"].astype(str),
            "driver_number": pd.to_numeric(laps["DriverNumber"], errors="coerce"),
            "lap_number": pd.to_numeric(laps["LapNumber"], errors="coerce"),
            "lap_duration": laps["LapTime"].dt.total_seconds(),
            "tyre_age": pd.to_numeric(laps["TyreLife"], errors="coerce"),
            "compound": laps["Compound"].astype(str).str.upper(),
            "stint_number": pd.to_numeric(laps["Stint"], errors="coerce"),
            "is_pit_out_lap": laps["PitOutTime"].notna().to_numpy(),
            "date_start": start + laps["LapStartTime"],
        }
    )
    frame["observed_at"] = frame["date_start"] + pd.to_timedelta(frame["lap_duration"], unit="s")
    # A lap is unusable unless the source itself vouches for it: deleted laps,
    # laps flagged inaccurate by the timing feed, in-laps and any lap run under a
    # non-green track status are dropped rather than modelled.
    frame["interrupted"] = (
        _flag(laps["Deleted"])
        | ~_flag(laps["IsAccurate"])
        | laps["PitInTime"].notna().to_numpy()
        | laps["TrackStatus"].astype(str).ne("1").to_numpy()
    )
    frame = frame.merge(codes, on="driver_code", how="left", validate="many_to_one")
    frame = frame.dropna(subset=["driver_id", "observed_at"]).sort_values("observed_at")
    weather = getattr(session, "weather_data", None)
    if weather is not None and not weather.empty:
        weather = pd.DataFrame(
            {
                "weather_at": start + weather["Time"],
                "rainfall": weather["Rainfall"].astype(float),
            }
        ).sort_values("weather_at")
        frame = pd.merge_asof(
            frame, weather, left_on="observed_at", right_on="weather_at", direction="backward"
        )
        frame["rainfall"] = frame["rainfall"].ffill().fillna(0.0)
    else:
        # No weather feed means no evidence of a dry session, so nothing is kept.
        frame["rainfall"] = 1.0
    frame["session_start"] = start
    frame["cutoff_utc"] = frame["observed_at"].max() + AVAILABILITY_MARGIN
    return frame.reset_index(drop=True)
