from __future__ import annotations

import hashlib
import time
from pathlib import Path
from urllib.parse import quote, urlencode

import pandas as pd

OPENF1 = "https://api.openf1.org/v1"
OPEN_METEO_PREVIOUS = "https://previous-runs-api.open-meteo.com/v1/forecast"
OPEN_METEO_HIST = "https://historical-forecast-api.open-meteo.com/v1/forecast"
TRACING_RAW = "https://raw.githubusercontent.com/TracingInsights/2026/main"


def openf1_url(endpoint: str, **params) -> str:
    return f"{OPENF1}/{quote(endpoint.strip('/'))}?{urlencode(params, doseq=True)}"


def tracing_url(event: str, session: str, filename: str) -> str:
    return f"{TRACING_RAW}/{quote(event)}/{quote(session)}/{quote(filename)}"


def openmeteo_previous_url(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    hourly: list[str],
    past_days: int | None = None,
) -> str:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly),
        "timezone": "UTC",
    }
    if past_days is not None:
        params["past_days"] = past_days
    return OPEN_METEO_PREVIOUS + "?" + urlencode(params)


def normalize_openf1_car_data(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    ren = {
        "date": "timestamp",
        "speed": "speed",
        "throttle": "throttle",
        "brake": "brake",
        "rpm": "rpm",
        "n_gear": "gear",
        "drs": "drs",
    }
    d = d.rename(columns=ren)
    for c in ["speed", "throttle", "brake", "rpm", "gear", "drs"]:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "timestamp" in d:
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    return d


def normalize_openf1_weather(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().rename(
        columns={
            "date": "timestamp",
            "air_temperature": "air_temp",
            "track_temperature": "track_temp",
            "wind_speed": "wind_speed",
            "wind_direction": "wind_direction",
            "rainfall": "rainfall",
            "humidity": "humidity",
            "pressure": "pressure",
        }
    )
    if "timestamp" in d:
        d.timestamp = pd.to_datetime(d.timestamp, utc=True, errors="coerce")
    return d


def normalize_tracing_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    lower = {c: c.strip().lower().replace(" ", "_") for c in d.columns}
    d = d.rename(columns=lower)
    aliases = {
        "distance_m": "distance",
        "speed_kmh": "speed",
        "throttle_pct": "throttle",
        "brake_pct": "brake",
        "x_m": "x",
        "y_m": "y",
        "z_m": "z",
        "time": "time_s",
    }
    d = d.rename(columns={k: v for k, v in aliases.items() if k in d.columns})
    return d


def http_get_cached(url: str, cache_dir: str | Path, timeout: int = 30, retries: int = 3) -> Path:
    """Optional runtime downloader. Uses requests only when called; all downloads are cached."""
    import requests

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()[:24]
    out = cache / f"{key}.bin"
    if out.exists() and out.stat().st_size > 0:
        return out
    err = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            out.write_bytes(r.content)
            return out
        except Exception as e:
            err = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"download failed: {url}") from err
