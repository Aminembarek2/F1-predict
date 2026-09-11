"""Cached OpenF1 practice ingestion; public historical data begins in 2023."""

from __future__ import annotations

import logging
from urllib.parse import urlencode

import pandas as pd

from ..config import OPENF1_BURST, OPENF1_MIN_INTERVAL_S
from ..http_cache import CachedSession
from . import normalise_practice

log = logging.getLogger(__name__)


def _utc(values: pd.Series) -> pd.Series:
    """Parse OpenF1 timestamps, which mix whole and fractional seconds.

    The feed emits both "...:28+00:00" and "...:28.079+00:00" in one response, so
    pandas infers a format from the first row and then fails partway down the
    column. ISO 8601 is declared explicitly instead of inferred.
    """
    return pd.to_datetime(values, utc=True, format="ISO8601")


class OpenF1:
    """One rate-limited connection with provenance for each response."""

    def __init__(self) -> None:
        self.http = CachedSession("openf1", OPENF1_MIN_INTERVAL_S, burst=OPENF1_BURST)

    def get(self, endpoint: str, **parameters: object) -> pd.DataFrame:
        url = f"https://api.openf1.org/v1/{endpoint}?{urlencode(parameters)}"
        payload = self.http.get_json(url)
        if not isinstance(payload, list):
            raise ValueError(f"unexpected {endpoint} response")
        frame = pd.DataFrame(payload)
        frame.attrs["source_url"] = url
        frame.attrs["fetched_at"] = self.http.record_for(url).fetched_at
        return frame


def load_practice(
    seasons: list[int], schedule: pd.DataFrame, results: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch FP2 laps with stints, identities, weather and race-control exclusions.

    Calendar matching uses session dates, not a hand-maintained circuit mapping.
    Missing sessions are reported, never silently replaced by synthetic data.
    """
    client, frames, coverage = OpenF1(), [], []
    for season in seasons:
        if season < 2023:
            coverage.append({"season": season, "status": "unavailable_before_2023"})
            continue
        sessions = client.get("sessions", year=season)
        fp2 = sessions.loc[sessions["session_name"].eq("Practice 2")]
        races = schedule.loc[schedule["season"].eq(season)].copy()
        dates = pd.to_datetime(races["date"], utc=True)
        code_map = results.loc[
            results["season"].eq(season), ["driver_code", "driver_id"]
        ].drop_duplicates()
        if code_map["driver_code"].duplicated().any():
            raise ValueError("ambiguous driver abbreviation within a season")
        for session in fp2.to_dict("records"):
            start = pd.Timestamp(session["date_start"])
            end = pd.Timestamp(session["date_end"])
            candidates = races.loc[
                (dates >= start.normalize())
                & (dates <= start.normalize() + pd.to_timedelta(3, unit="D"))
            ]
            if len(candidates) != 1:
                coverage.append(
                    {
                        "season": season,
                        "session_key": session["session_key"],
                        "status": "calendar_match_failed",
                    }
                )
                continue
            race_id = candidates["race_id"].iloc[0]
            log.info("practice ingestion %s", race_id)
            key = int(session["session_key"])
            laps = client.get("laps", session_key=key)
            stints = client.get("stints", session_key=key)
            drivers = client.get("drivers", session_key=key)
            control = client.get("race_control", session_key=key)
            weather = client.get("weather", session_key=key)
            if any(frame.empty for frame in (laps, stints, drivers, weather)):
                coverage.append({"race_id": race_id, "status": "incomplete_practice_inputs"})
                continue
            driver_map = (
                drivers[["driver_number", "name_acronym"]]
                .drop_duplicates()
                .merge(
                    code_map,
                    left_on="name_acronym",
                    right_on="driver_code",
                    how="left",
                    validate="one_to_one",
                )
            )
            source = laps.attrs["source_url"]
            fetched = max(
                frame.attrs["fetched_at"] for frame in (laps, stints, drivers, control, weather)
            )
            laps = laps.merge(
                driver_map[["driver_number", "driver_id"]],
                on="driver_number",
                validate="many_to_one",
            )
            laps["observed_at"] = _utc(laps["date_start"]) + pd.to_timedelta(
                laps["lap_duration"], unit="s"
            )
            laps["tyre_age"], laps["compound"], laps["stint_number"] = (
                float("nan"),
                "",
                float("nan"),
            )
            for stint in stints.to_dict("records"):
                mask = laps["driver_number"].eq(stint["driver_number"]) & laps[
                    "lap_number"
                ].between(stint["lap_start"], stint["lap_end"])
                laps.loc[mask, "compound"] = stint["compound"]
                laps.loc[mask, "stint_number"] = stint["stint_number"]
                laps.loc[mask, "tyre_age"] = (
                    laps.loc[mask, "lap_number"]
                    - stint["lap_start"]
                    + stint.get("tyre_age_at_start", 0)
                )
            # Conservatively discard laps overlapping any race-control message.
            # This includes interrupted/deleted laps; it can discard extra clean
            # laps, but must not invent a clean flag when control is missing.
            laps["interrupted"] = control.empty
            if not control.empty:
                for date in _utc(control["date"]):
                    laps.loc[
                        (_utc(laps["date_start"]) <= date)
                        & (laps["observed_at"] + pd.to_timedelta(2, unit="m") >= date),
                        "interrupted",
                    ] = True
            laps = laps.dropna(subset=["observed_at"]).sort_values("observed_at")
            weather["weather_at"] = _utc(weather["date"])
            laps = pd.merge_asof(
                laps,
                weather[["weather_at", "rainfall"]].sort_values("weather_at"),
                left_on="observed_at",
                right_on="weather_at",
                direction="backward",
            )
            laps["race_id"], laps["season"], laps["stage"] = race_id, season, "POST_FP2"
            laps["session_start"] = start
            laps["cutoff_utc"] = end + pd.to_timedelta(5, unit="m")
            laps["source"], laps["fetched_at"] = source, fetched
            frames.append(laps)
            coverage.append({"race_id": race_id, "status": "downloaded", "laps": len(laps)})
    laps = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return normalise_practice(laps), pd.DataFrame(coverage)


def _optional(client: OpenF1, endpoint: str, **parameters: object) -> pd.DataFrame:
    """Fetch a collection that some sessions simply do not have."""
    try:
        return client.get(endpoint, **parameters)
    except FileNotFoundError:
        log.info("%s unavailable for %s", endpoint, parameters)
        return pd.DataFrame()


def load_race_events(
    seasons: list[int], schedule: pd.DataFrame, results: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch on-track overtakes and pit-lane times for completed races.

    These are the two quantities the forecaster currently guesses at rather than
    measures: overtaking difficulty is inferred from how much the finishing order
    resembles the grid, and the strategy variance in the simulator is a
    hand-chosen constant. Both are observable directly.

    Returns ``(overtakes, pit, coverage)``. Historical coverage starts in 2023.
    """
    client, passes, stops, coverage = OpenF1(), [], [], []
    for season in seasons:
        if season < 2023:
            coverage.append({"season": season, "status": "unavailable_before_2023"})
            continue
        sessions = client.get("sessions", year=season, session_name="Race")
        if sessions.empty:
            coverage.append({"season": season, "status": "no_race_sessions"})
            continue
        races = schedule.loc[schedule["season"].eq(season)].copy()
        dates = pd.to_datetime(races["date"], utc=True)
        for session in sessions.to_dict("records"):
            start = _utc(pd.Series([session["date_start"]])).iloc[0]
            match = races.loc[(dates - start).abs() <= pd.to_timedelta(1, unit="D")]
            if len(match) != 1:
                coverage.append({"season": season, "status": "calendar_match_failed"})
                continue
            race_id = match["race_id"].iloc[0]
            key = int(session["session_key"])
            log.info("race events %s", race_id)
            # Endpoint coverage varies by session: the older races predate some
            # collections entirely and answer 404. A missing collection is a gap
            # in that race's record, not a reason to abandon the calendar.
            overtakes = _optional(client, "overtakes", session_key=key)
            pit = _optional(client, "pit", session_key=key)
            if not overtakes.empty:
                overtakes["race_id"], overtakes["season"] = race_id, season
                overtakes["circuit_id"] = match["circuit_id"].iloc[0]
                passes.append(overtakes)
            if not pit.empty:
                pit["race_id"], pit["season"] = race_id, season
                pit["circuit_id"] = match["circuit_id"].iloc[0]
                stops.append(pit)
            coverage.append(
                {
                    "race_id": race_id,
                    "status": "downloaded",
                    "overtakes": len(overtakes),
                    "pit_stops": len(pit),
                }
            )

    def frame(rows: list[pd.DataFrame]) -> pd.DataFrame:
        # Empty frames are dropped first: concatenating them makes pandas infer
        # column dtypes from nothing, which it warns about and will stop doing.
        rows = [r for r in rows if not r.empty]
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    return frame(passes), frame(stops), pd.DataFrame(coverage)
