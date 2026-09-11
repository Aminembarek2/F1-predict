"""Jolpica-F1 adapter (Ergast-compatible historical results).

Provides the leakage-relevant backbone: race calendar, classified results,
qualifying times and grid positions. Every frame is returned tidy with explicit
dtypes; no downstream module parses raw JSON.

Availability note
-----------------
``load_results`` returns *outcomes*. A result row for race R is only legal as a
feature for races strictly after R. :mod:`f1pred.leakage` enforces this.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import pandas as pd

from ..config import JOLPICA_MIN_INTERVAL_S, regime_for
from ..http_cache import CachedSession

log = logging.getLogger(__name__)

BASE = "https://api.jolpi.ca/ergast/f1"
PAGE_SIZE = 100

_session = CachedSession("jolpica", JOLPICA_MIN_INTERVAL_S)


def _paged(
    path: str, table_key: str, list_key: str, *, refresh: bool = False
) -> list[dict[str, Any]]:
    """Follow Ergast pagination until every record for ``path`` is collected."""
    out: list[dict[str, Any]] = []
    offset = 0
    while True:
        url = f"{BASE}/{path}.json?limit={PAGE_SIZE}&offset={offset}"
        payload = _session.get_json(url, refresh=refresh)["MRData"]
        table = payload[table_key]
        records = table[list_key]
        fetched_at = _session.record_for(url).fetched_at
        for record in records:
            record["_fetched_at"] = fetched_at
        out.extend(records)
        total = int(payload["total"])
        offset += PAGE_SIZE
        if offset >= total or not records:
            return out


def _mmss_to_seconds(value: str | None) -> float:
    """Parse ``m:ss.sss`` / ``ss.sss`` lap-time strings into seconds."""
    if not value or not isinstance(value, str):
        return float("nan")
    parts = value.strip().split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return float("nan")
    return float("nan")


def load_schedule(seasons: Iterable[int], *, refresh: bool = False) -> pd.DataFrame:
    """Race calendar with UTC start timestamps and circuit identity."""
    rows: list[dict[str, Any]] = []
    for season in seasons:
        for race in _paged(f"{season}/races", "RaceTable", "Races", refresh=refresh):
            circuit = race["Circuit"]
            location = circuit["Location"]
            rows.append(
                {
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "race_name": race["raceName"],
                    "circuit_id": circuit["circuitId"],
                    "circuit_name": circuit["circuitName"],
                    "locality": location.get("locality"),
                    "country": location.get("country"),
                    "lat": pd.to_numeric(location.get("lat"), errors="coerce"),
                    "lon": pd.to_numeric(location.get("long"), errors="coerce"),
                    "date": race["date"],
                    "time": race.get("time"),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["race_start_utc"] = pd.to_datetime(
        df["date"] + " " + df["time"].fillna("12:00:00Z").str.replace("Z", "", regex=False),
        utc=True,
        errors="coerce",
    )
    df["regime"] = df["season"].map(regime_for)
    df["race_id"] = df["season"].astype(str) + "_" + df["round"].astype(str).str.zfill(2)
    return df.sort_values(["season", "round"]).reset_index(drop=True)


# Statuses that describe a car which reached the flag. Jolpica uses "Lapped" for
# recent seasons where classic Ergast used "+1 Lap", so both spellings appear.
_FINISHED_STATUSES = frozenset({"Finished", "Lapped"})
_INCIDENT_WORDS = (
    "collision",
    "accident",
    "spun",
    "damage",
    "puncture",
    "crash",
    "contact",
)
_NON_STARTER_STATUSES = frozenset(
    {"Did not start", "Did not qualify", "Did not prequalify", "Withdrew", "Injury"}
)


def is_finished_status(status: str) -> bool:
    """True when the status string describes a car that completed the race."""
    s = (status or "").strip()
    return s in _FINISHED_STATUSES or s.startswith("+")


def classify_status(status: str, position_text: str) -> str:
    """Classify one result into ``finished`` / ``incident`` / ``mechanical`` / ``non_starter``.

    ``position_text`` is the authority on *whether* the car was classified: a
    numeric value means the FIA gave it a finishing position (it covered at least
    90% of the winner's distance), while ``R``/``D``/``E``/``W``/``F``/``N`` mean
    it was not. The ``status`` string is used only to attribute the *cause*.

    This split matters: "Lapped" and "+1 Lap" are classified finishes, whereas a
    generic "Retired" may be either classified or not depending on distance run.
    """
    s = (status or "").strip()
    pt = (position_text or "").strip()
    if s in _NON_STARTER_STATUSES or pt in {"W", "F"}:
        return "non_starter"
    if pt.isdigit():
        return "finished"
    low = s.lower()
    if any(w in low for w in _INCIDENT_WORDS) or pt == "D":
        return "incident"
    if is_finished_status(s):
        # Not classified despite a "finished" status: treat as an unspecified retirement.
        return "mechanical"
    return "mechanical"


def load_results(seasons: Iterable[int], *, refresh: bool = False) -> pd.DataFrame:
    """Classified race results, one row per driver per race."""
    rows: list[dict[str, Any]] = []
    for season in seasons:
        for race in _paged(f"{season}/results", "RaceTable", "Races", refresh=refresh):
            race_id = f"{int(race['season'])}_{int(race['round']):02d}"
            for res in race["Results"]:
                driver = res["Driver"]
                constructor = res["Constructor"]
                fastest = res.get("FastestLap") or {}
                rows.append(
                    {
                        "race_id": race_id,
                        "observed_at": race["_fetched_at"],
                        "season": int(race["season"]),
                        "round": int(race["round"]),
                        "circuit_id": race["Circuit"]["circuitId"],
                        "date": race["date"],
                        "driver_id": driver["driverId"],
                        "driver_code": driver.get("code"),
                        "driver_name": f"{driver['givenName']} {driver['familyName']}",
                        "dob": driver.get("dateOfBirth"),
                        "constructor_id": constructor["constructorId"],
                        "constructor_name": constructor["name"],
                        "grid": pd.to_numeric(res.get("grid"), errors="coerce"),
                        "finish_position": pd.to_numeric(
                            res.get("position"), errors="coerce"
                        ),  # == positionOrder
                        "position_text": res.get("positionText"),
                        "points": pd.to_numeric(res.get("points"), errors="coerce"),
                        "laps": pd.to_numeric(res.get("laps"), errors="coerce"),
                        "status": res.get("status"),
                        "total_ms": pd.to_numeric(
                            (res.get("Time") or {}).get("millis"), errors="coerce"
                        ),
                        "fastest_lap_s": _mmss_to_seconds((fastest.get("Time") or {}).get("time")),
                        "fastest_lap_rank": pd.to_numeric(fastest.get("rank"), errors="coerce"),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # ``position`` in the Ergast schema is the full classification *order* (1..N),
    # present for every entry including retirements. It is the ranking target.
    df = df.rename(columns={"finish_position": "position_order"})
    df["status_class"] = [
        classify_status(s, p) for s, p in zip(df["status"], df["position_text"], strict=True)
    ]
    df["classified"] = df["position_text"].str.isdigit().fillna(False).astype(int)
    df["started"] = (df["status_class"] != "non_starter").astype(int)
    # A DNF is a car that started but was not classified; that is what removes a
    # driver from the points, and it is the quantity the simulator samples.
    df["dnf"] = ((df["started"] == 1) & (df["classified"] == 0)).astype(int)
    df["dnf_mechanical"] = ((df["dnf"] == 1) & (df["status_class"] == "mechanical")).astype(int)
    df["dnf_incident"] = ((df["dnf"] == 1) & (df["status_class"] == "incident")).astype(int)
    df["regime"] = df["season"].map(regime_for)
    # Field size counts starters only; non-starters are dropped by the feature layer.
    df["field_size"] = df.groupby("race_id")["started"].transform("sum")
    df["winner"] = ((df["position_order"] == 1) & (df["classified"] == 1)).astype(int)
    df["podium"] = ((df["position_order"] <= 3) & (df["classified"] == 1)).astype(int)
    df["top10"] = ((df["position_order"] <= 10) & (df["classified"] == 1)).astype(int)
    return df.sort_values(["season", "round", "position_order"]).reset_index(drop=True)


def load_qualifying(seasons: Iterable[int], *, refresh: bool = False) -> pd.DataFrame:
    """Qualifying sessions with per-segment times and the best lap in seconds."""
    rows: list[dict[str, Any]] = []
    for season in seasons:
        for race in _paged(f"{season}/qualifying", "RaceTable", "Races", refresh=refresh):
            race_id = f"{int(race['season'])}_{int(race['round']):02d}"
            for q in race["QualifyingResults"]:
                q1, q2, q3 = (_mmss_to_seconds(q.get(k)) for k in ("Q1", "Q2", "Q3"))
                best = min([t for t in (q1, q2, q3) if t == t], default=float("nan"))
                rows.append(
                    {
                        "race_id": race_id,
                        "season": int(race["season"]),
                        "round": int(race["round"]),
                        "driver_id": q["Driver"]["driverId"],
                        "constructor_id": q["Constructor"]["constructorId"],
                        "qual_position": pd.to_numeric(q.get("position"), errors="coerce"),
                        "observed_at": race["_fetched_at"],
                        "q1_s": q1,
                        "q2_s": q2,
                        "q3_s": q3,
                        "qual_best_s": best,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Percentage gap to the session pole is the only cross-circuit comparable form.
    pole = df.groupby("race_id")["qual_best_s"].transform("min")
    df["qual_gap_pct"] = 100.0 * (df["qual_best_s"] - pole) / pole
    return df.sort_values(["season", "round", "qual_position"]).reset_index(drop=True)


def load_driver_standings(season: int, round_no: int, *, refresh: bool = False) -> pd.DataFrame:
    """Championship standings *after* a completed round (pre-weekend legal for round+1)."""
    url = f"{BASE}/{season}/{round_no}/driverStandings.json?limit=50"
    lists = _session.get_json(url, refresh=refresh)["MRData"]["StandingsTable"]["StandingsLists"]
    if not lists:
        return pd.DataFrame()
    rows = [
        {
            "season": season,
            "after_round": round_no,
            "driver_id": s["Driver"]["driverId"],
            "constructor_id": s["Constructors"][0]["constructorId"],
            "championship_position": pd.to_numeric(s.get("position"), errors="coerce"),
            "position_text": s.get("positionText"),
            "observed_at": _session.record_for(url).fetched_at,
            "championship_points": float(s["points"]),
            "championship_wins": int(s["wins"]),
        }
        for s in lists[0]["DriverStandings"]
    ]
    return pd.DataFrame(rows)
