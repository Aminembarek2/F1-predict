from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import (
    corner_metrics,
    curvature_from_xy,
    practice_pace,
    resample_telemetry,
    segment_corners,
    track_evolution_correct,
)


def fastf1_session(year: int, event, session_code: str, cache_dir: str | Path):
    import fastf1

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    s = fastf1.get_session(year, event, session_code)
    s.load(telemetry=True, weather=True, messages=False)
    return s


def fastf1_corners(session) -> pd.DataFrame:
    ci = session.get_circuit_info()
    c = ci.corners.copy()
    ren = {"Number": "corner", "Distance": "distance", "Angle": "angle"}
    c = c.rename(columns={k: v for k, v in ren.items() if k in c.columns})
    keep = [x for x in ["corner", "distance", "angle"] if x in c.columns]
    return c[keep].copy()


def lap_telemetry_table(lap) -> pd.DataFrame:
    tel = lap.get_telemetry().copy()
    ren = {
        "Distance": "distance",
        "Speed": "speed",
        "Throttle": "throttle",
        "Brake": "brake",
        "RPM": "rpm",
        "nGear": "gear",
        "DRS": "drs",
        "X": "x",
        "Y": "y",
        "Z": "z",
        "Time": "time_delta",
    }
    tel = tel.rename(columns={k: v for k, v in ren.items() if k in tel.columns})
    if "time_delta" in tel:
        tel["time_s"] = tel.time_delta.dt.total_seconds()
    for c in ["distance", "speed", "throttle", "brake", "rpm", "gear", "drs", "x", "y", "z"]:
        if c in tel:
            tel[c] = pd.to_numeric(tel[c], errors="coerce")
    return tel


def driver_corner_features(
    session, driver, lap_selector="fastest", distance_step_m=2.0
) -> pd.DataFrame:
    laps = session.laps.pick_drivers(driver)
    if laps.empty:
        return pd.DataFrame()
    lap = laps.pick_fastest() if lap_selector == "fastest" else laps.iloc[int(lap_selector)]
    tel = lap_telemetry_table(lap)
    tel = tel.dropna(subset=["distance", "speed"])
    if tel.empty:
        return pd.DataFrame()
    tel = resample_telemetry(tel, distance_step_m)
    if {"x", "y"}.issubset(tel.columns):
        tel["curvature"] = curvature_from_xy(tel.x, tel.y)
    corners = fastf1_corners(session)
    seg = segment_corners(tel, corners, before_m=120, after_m=180)
    rows = []
    distances = corners.sort_values("distance").distance.to_numpy(float)
    lap_len = float(tel.distance.max())
    for cid, g in seg.groupby("corner"):
        center = float(g.corner_distance.iloc[0])
        # downstream straight length approximated to next official corner marker
        later = distances[distances > center]
        next_d = float(later.min()) if len(later) else lap_len
        m = corner_metrics(g, max(0, next_d - center))
        m.update({"driver": str(driver), "corner": cid, "corner_distance": center})
        rows.append(m)
    return pd.DataFrame(rows)


def session_corner_dataset(session, drivers=None) -> pd.DataFrame:
    drivers = drivers or list(session.laps.Driver.dropna().astype(str).unique())
    frames = []
    for d in drivers:
        try:
            x = driver_corner_features(session, d)
            if len(x):
                frames.append(x)
        except Exception:
            continue
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out):
        # Within-corner normalized performance: lower corner time is ideal, but when only metrics exist,
        # combine normalized apex+exit speed as a transparent first proxy.
        out["corner_performance"] = (
            out.groupby("corner")["apex_speed_kmh"].transform(
                lambda s: (s - s.mean()) / (s.std(ddof=0) or 1)
            )
            + out.groupby("corner")["exit_speed_kmh"].transform(
                lambda s: (s - s.mean()) / (s.std(ddof=0) or 1)
            )
        ) / 2
    return out


def session_practice_summary(session) -> pd.DataFrame:
    laps = session.laps.copy()
    if laps.empty:
        return pd.DataFrame()
    d = pd.DataFrame(
        {
            "driver": laps.Driver.astype(str),
            "lap_time": laps.LapTime.dt.total_seconds(),
            "session_seconds": laps.Time.dt.total_seconds(),
            "stint": pd.to_numeric(laps.Stint, errors="coerce") if "Stint" in laps else np.nan,
            "tyre_age": pd.to_numeric(laps.TyreLife, errors="coerce")
            if "TyreLife" in laps
            else np.nan,
            "compound": laps.Compound.astype(str) if "Compound" in laps else "",
        }
    )
    d["corrected_lap"] = track_evolution_correct(d, "session_seconds", "lap_time")
    return practice_pace(d, "driver", "corrected_lap", "stint")


def fastf1_weather(session) -> pd.DataFrame:
    w = session.weather_data.copy()
    if w.empty:
        return w
    ren = {
        "Time": "time_delta",
        "AirTemp": "air_temp",
        "TrackTemp": "track_temp",
        "Humidity": "humidity",
        "Pressure": "pressure",
        "Rainfall": "rainfall",
        "WindDirection": "wind_direction",
        "WindSpeed": "wind_speed",
    }
    w = w.rename(columns={k: v for k, v in ren.items() if k in w.columns})
    if "time_delta" in w:
        base = (
            pd.Timestamp(session.date, tz="UTC")
            if pd.Timestamp(session.date).tzinfo is None
            else pd.Timestamp(session.date).tz_convert("UTC")
        )
        w["timestamp"] = base + pd.to_timedelta(w.time_delta)
    return w
