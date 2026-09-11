from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

EPS = 1e-9


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def curvature_from_xy(x: Sequence[float], y: Sequence[float]) -> np.ndarray:
    """Planar curvature kappa = |x'y'' - y'x''|/(x'^2+y'^2)^(3/2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 5:
        return np.full(len(x), np.nan)
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = np.power(dx * dx + dy * dy, 1.5)
    k = np.abs(dx * ddy - dy * ddx) / np.where(denom > EPS, denom, np.nan)
    return k


def curvature_indices(curvature: Sequence[float]) -> dict:
    k = np.asarray(curvature, dtype=float)
    k = k[np.isfinite(k)]
    if not len(k):
        return {
            "curvature_exposure": np.nan,
            "curvature_severity": np.nan,
            "curvature_variation": np.nan,
        }
    return {
        "curvature_exposure": float(np.nanmean(np.abs(k))),
        "curvature_severity": float(np.nanpercentile(np.abs(k), 95)),
        "curvature_variation": float(np.nanstd(k)),
    }


def segment_corners(
    telemetry: pd.DataFrame,
    corner_markers: pd.DataFrame,
    before_m: float = 120.0,
    after_m: float = 180.0,
) -> pd.DataFrame:
    """Assign each telemetry row to the nearest official corner window by lap distance."""
    t = telemetry.copy()
    if "distance" not in t:
        raise ValueError("telemetry must contain distance")
    markers = corner_markers.copy()
    if "distance" not in markers:
        raise ValueError("corner_markers must contain distance")
    markers = markers.sort_values("distance").reset_index(drop=True)
    out = []
    for i, m in markers.iterrows():
        cid = m.get("corner", m.get("number", i + 1))
        center = float(m["distance"])
        g = t[(t.distance >= center - before_m) & (t.distance <= center + after_m)].copy()
        if g.empty:
            continue
        g["corner"] = cid
        g["corner_distance"] = center
        g["relative_distance"] = g.distance - center
        out.append(g)
    return (
        pd.concat(out, ignore_index=True)
        if out
        else pd.DataFrame(
            columns=list(t.columns) + ["corner", "corner_distance", "relative_distance"]
        )
    )


def braking_efficiency(
    entry_speed_kmh: float, apex_speed_kmh: float, braking_distance_m: float
) -> float:
    """Average deceleration magnitude implied by v^2=u^2+2as, in m/s^2."""
    if braking_distance_m is None or braking_distance_m <= 0:
        return np.nan
    u = float(entry_speed_kmh) / 3.6
    v = float(apex_speed_kmh) / 3.6
    return max(0.0, (u * u - v * v) / (2.0 * braking_distance_m))


def traction_index(apex_speed_kmh: float, exit_speed_kmh: float, delta_t_s: float) -> float:
    """Average longitudinal acceleration after apex in m/s^2."""
    if delta_t_s is None or delta_t_s <= 0:
        return np.nan
    return ((float(exit_speed_kmh) - float(apex_speed_kmh)) / 3.6) / delta_t_s


def exit_amplification(delta_exit_speed_kmh: float, following_straight_m: float) -> float:
    """Simple interaction score: exit-speed advantage × downstream straight length."""
    return (float(delta_exit_speed_kmh) / 3.6) * float(following_straight_m)


def straight_time_gain(
    exit_speed_a_kmh: float, exit_speed_b_kmh: float, straight_m: float
) -> float:
    """First-order constant-speed time advantage: positive means A faster than B."""
    va = max(float(exit_speed_a_kmh) / 3.6, EPS)
    vb = max(float(exit_speed_b_kmh) / 3.6, EPS)
    return float(straight_m) * (1.0 / vb - 1.0 / va)


def lateral_g(speed_kmh: float, curvature_1pm: float) -> float:
    v = float(speed_kmh) / 3.6
    return (v * v * float(curvature_1pm)) / 9.80665


def aero_commitment_index(speed_kmh: float, curvature_1pm: float, throttle_pct: float) -> float:
    """Telemetry proxy for high-speed aero/confidence: lateral-g × throttle commitment."""
    return lateral_g(speed_kmh, curvature_1pm) * np.clip(float(throttle_pct) / 100.0, 0, 1)


def wind_components(
    wind_speed: float, wind_direction_deg: float, track_heading_deg: float
) -> tuple[float, float]:
    """Return signed headwind and absolute crosswind in same speed units as wind_speed."""
    theta = np.deg2rad(float(wind_direction_deg) - float(track_heading_deg))
    head = float(wind_speed) * math.cos(theta)
    cross = abs(float(wind_speed) * math.sin(theta))
    return head, cross


def corner_metrics(segment: pd.DataFrame, following_straight_m: float = 0.0) -> dict:
    """Extract braking/apex/exit metrics from one corner telemetry window."""
    g = segment.sort_values("distance").copy()
    if g.empty:
        return {}
    speed = _num(g["speed"])
    dist = _num(g["distance"])
    apex_i = speed.idxmin()
    apex_speed = float(speed.loc[apex_i])
    apex_dist = float(dist.loc[apex_i])
    entry = g[g.distance <= apex_dist]
    exitg = g[g.distance >= apex_dist]
    entry_speed = float(_num(entry.speed).max()) if len(entry) else apex_speed
    exit_speed = float(_num(exitg.speed).iloc[-1]) if len(exitg) else apex_speed
    if "brake" in g:
        b = _num(entry.brake).fillna(0)
        active = entry[b > 0.5]
        brake_dist = max(1.0, apex_dist - float(active.distance.min())) if len(active) else np.nan
    else:
        brake_dist = np.nan
    if "time_s" in g:
        t0 = float(_num(g.loc[apex_i:apex_i, "time_s"]).iloc[0])
        tend = float(_num(exitg.time_s).iloc[-1])
        dt = max(tend - t0, EPS)
    else:
        avg = max((apex_speed + exit_speed) / 2 / 3.6, EPS)
        dt = max((float(exitg.distance.iloc[-1]) - apex_dist) / avg, EPS)
    kappa = float(np.nanmedian(_num(g.get("curvature", pd.Series(np.nan, index=g.index)))))
    throttle = float(np.nanmean(_num(g.get("throttle", pd.Series(np.nan, index=g.index)))))
    return {
        "entry_speed_kmh": entry_speed,
        "apex_speed_kmh": apex_speed,
        "exit_speed_kmh": exit_speed,
        "braking_distance_m": brake_dist,
        "braking_efficiency_ms2": braking_efficiency(entry_speed, apex_speed, brake_dist),
        "traction_index_ms2": traction_index(apex_speed, exit_speed, dt),
        "aero_commitment": aero_commitment_index(apex_speed, kappa, throttle)
        if np.isfinite(kappa) and np.isfinite(throttle)
        else np.nan,
        "exit_amplification": exit_amplification(exit_speed - apex_speed, following_straight_m),
    }


def track_demand_vector(corner_table: pd.DataFrame) -> pd.Series:
    """Compact circuit demand vector from corner-level telemetry/geometry."""
    c = corner_table.copy()
    s = _num(c["apex_speed_kmh"])
    n = max(len(c), 1)
    return pd.Series(
        {
            "low_speed_share": float((s < 140).sum() / n),
            "medium_speed_share": float(((s >= 140) & (s < 220)).sum() / n),
            "high_speed_share": float((s >= 220).sum() / n),
            "braking_demand": float(np.nanmean(_num(c.get("braking_efficiency_ms2", np.nan)))),
            "traction_demand": float(np.nanmean(_num(c.get("traction_index_ms2", np.nan)))),
            "aero_demand": float(np.nanmean(_num(c.get("aero_commitment", np.nan)))),
            "exit_importance": float(np.nanmean(_num(c.get("exit_amplification", np.nan)))),
        }
    )


def shrink_mean(values: Sequence[float], prior: float, strength: float = 5.0) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return float(prior)
    return float((x.sum() + strength * prior) / (len(x) + strength))


def car_capability_vector(
    corner_rows: pd.DataFrame, prior: pd.Series | None = None, shrinkage: float = 8.0
) -> pd.Series:
    """Infer capability dimensions from car telemetry with empirical-Bayes shrinkage."""
    c = corner_rows.copy()
    prior = prior if prior is not None else pd.Series(dtype=float)
    dims = {
        "low_speed": _num(c.loc[_num(c.apex_speed_kmh) < 140, "corner_performance"]),
        "medium_speed": _num(
            c.loc[
                (_num(c.apex_speed_kmh) >= 140) & (_num(c.apex_speed_kmh) < 220),
                "corner_performance",
            ]
        ),
        "high_speed": _num(c.loc[_num(c.apex_speed_kmh) >= 220, "corner_performance"]),
        "braking": _num(c.get("braking_efficiency_ms2", pd.Series(dtype=float))),
        "traction": _num(c.get("traction_index_ms2", pd.Series(dtype=float))),
        "aero": _num(c.get("aero_commitment", pd.Series(dtype=float))),
    }
    out = {}
    for k, v in dims.items():
        p = float(prior.get(k, np.nanmedian(v) if len(v) else 0.0))
        out[k] = shrink_mean(v, p, shrinkage)
    return pd.Series(out)


def cosine_compatibility(track_vec: Sequence[float], car_vec: Sequence[float]) -> float:
    a = np.asarray(track_vec, dtype=float)
    b = np.asarray(car_vec, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return np.nan
    a = a[mask]
    b = b[mask]
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > EPS else np.nan


def nearest_tracks(
    target: pd.Series, historical: pd.DataFrame, feature_cols: list[str], n: int = 5
) -> pd.DataFrame:
    rows = []
    tv = target[feature_cols].to_numpy(float)
    for idx, r in historical.iterrows():
        rows.append((idx, cosine_compatibility(tv, r[feature_cols].to_numpy(float))))
    return (
        pd.DataFrame(rows, columns=["index", "similarity"])
        .sort_values("similarity", ascending=False)
        .head(n)
    )


def teammate_adjusted_decomposition(
    df: pd.DataFrame,
    y_col: str = "performance",
    driver_col: str = "driver",
    car_col: str = "car",
    alpha: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ridge fixed-effects decomposition into car and teammate-adjusted driver effects."""
    d = df[[y_col, driver_col, car_col]].dropna().copy()
    X = d[[driver_col, car_col]]
    y = _num(d[y_col]).to_numpy()
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ct = ColumnTransformer([("cat", enc, [driver_col, car_col])], remainder="drop")
    model = make_pipeline(ct, Ridge(alpha=alpha, fit_intercept=True))
    model.fit(X, y)
    names = model.named_steps["columntransformer"].get_feature_names_out()
    coef = model.named_steps["ridge"].coef_
    effects = pd.DataFrame({"feature": names, "effect": coef})
    drivers = effects[effects.feature.str.contains(driver_col + "_")].copy()
    cars = effects[effects.feature.str.contains(car_col + "_")].copy()
    return drivers, cars


def temperature_sensitivity(
    df: pd.DataFrame, pace_col="pace_residual", temp_col="track_temp"
) -> float:
    d = df[[pace_col, temp_col]].dropna()
    if len(d) < 3:
        return np.nan
    X = d[[temp_col]].to_numpy()
    y = d[pace_col].to_numpy()
    return float(HuberRegressor(max_iter=1000).fit(X, y).coef_[0])


def wet_skill_residual(
    df: pd.DataFrame, driver_col="driver", car_col="car", pace_col="pace_residual", wet_col="wet"
) -> pd.Series:
    d = df[df[wet_col].astype(bool)].copy()
    if d.empty:
        return pd.Series(dtype=float)
    car_mean = d.groupby(car_col)[pace_col].mean()
    d["car_expected"] = d[car_col].map(car_mean)
    # Lower pace residual is better, so invert residual advantage.
    return -(d[pace_col] - d.car_expected).groupby(d[driver_col]).mean()


def tyre_degradation(
    df: pd.DataFrame,
    lap_age_col="tyre_age",
    lap_time_col="lap_time",
    temp_col: str | None = "track_temp",
) -> dict:
    cols = [lap_age_col, lap_time_col] + ([temp_col] if temp_col and temp_col in df else [])
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 5:
        return {"linear_deg": np.nan, "quadratic_deg": np.nan, "temp_effect": np.nan}
    age = d[lap_age_col].to_numpy()
    X = [age, age**2]
    if temp_col and temp_col in d:
        X.append(d[temp_col].to_numpy())
    X = np.column_stack(X)
    y = d[lap_time_col].to_numpy()
    m = HuberRegressor(max_iter=1000).fit(X, y)
    return {
        "linear_deg": float(m.coef_[0]),
        "quadratic_deg": float(m.coef_[1]),
        "temp_effect": float(m.coef_[2]) if X.shape[1] > 2 else np.nan,
    }


def tyre_warmup_laps(lap_times: Sequence[float], tolerance_s: float = 0.2) -> float:
    x = np.asarray(lap_times, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    steady = float(np.median(x[max(2, len(x) // 2) :]))
    for i, v in enumerate(x):
        if v <= steady + tolerance_s:
            return float(i + 1)
    return float(len(x))


def fuel_correct(
    lap_times: Sequence[float], lap_numbers: Sequence[float], fuel_slope_s_per_lap: float = -0.035
) -> np.ndarray:
    lt = np.asarray(lap_times, dtype=float)
    lap = np.asarray(lap_numbers, dtype=float)
    # Remove expected fuel-burn improvement to put laps on common fuel basis.
    return lt - fuel_slope_s_per_lap * (lap - np.nanmin(lap))


def track_evolution_correct(
    df: pd.DataFrame, time_col="session_seconds", lap_time_col="lap_time", bins: int = 8
) -> pd.Series:
    d = df[[time_col, lap_time_col]].copy()
    t = _num(d[time_col])
    y = _num(d[lap_time_col])
    valid = t.notna() & y.notna()
    if valid.sum() < 4:
        return y
    q = pd.qcut(t[valid], q=min(bins, valid.sum()), duplicates="drop")
    med = y[valid].groupby(q, observed=True).median()
    mapping = {k: v for k, v in med.items()}
    baseline = float(np.nanmin(list(mapping.values())))
    corr = pd.Series(np.nan, index=df.index, dtype=float)
    for idx, binv in q.items():
        corr.loc[idx] = y.loc[idx] - (mapping[binv] - baseline)
    return corr


def practice_pace(
    df: pd.DataFrame, driver_col="driver", lap_time_col="corrected_lap", stint_col="stint"
) -> pd.DataFrame:
    d = df.copy()
    d[lap_time_col] = _num(d[lap_time_col])
    rows = []
    for drv, g in d.groupby(driver_col):
        clean = g[lap_time_col].dropna().sort_values()
        one = float(clean.head(min(3, len(clean))).median()) if len(clean) else np.nan
        long = []
        if stint_col in g:
            for _, s in g.groupby(stint_col):
                vals = s[lap_time_col].dropna()
                if len(vals) >= 5:
                    long.append(float(vals.median()))
        rows.append(
            {
                driver_col: drv,
                "single_lap_pace": one,
                "long_run_pace": float(np.mean(long)) if long else np.nan,
            }
        )
    return pd.DataFrame(rows)


def traffic_class(gap_to_car_ahead_s: float) -> str:
    if not np.isfinite(gap_to_car_ahead_s):
        return "clean"
    if gap_to_car_ahead_s < 1.0:
        return "dirty_lt1"
    if gap_to_car_ahead_s < 2.0:
        return "dirty_1_2"
    return "clean"


def dirty_air_penalty(df: pd.DataFrame, lap_time_col="pace_residual", gap_col="gap_ahead") -> float:
    d = df[[lap_time_col, gap_col]].dropna().copy()
    d["traffic"] = d[gap_col].map(traffic_class)
    clean = _num(d.loc[d.traffic == "clean", lap_time_col])
    dirty = _num(d.loc[d.traffic != "clean", lap_time_col])
    return float(dirty.mean() - clean.mean()) if len(clean) and len(dirty) else np.nan


def fit_overtake_model(df: pd.DataFrame, target="passed", features: list[str] | None = None):
    features = features or ["gap", "pace_delta", "tyre_delta", "straight_length", "drs_available"]
    d = df[features + [target]].apply(pd.to_numeric, errors="coerce").dropna()
    if d[target].nunique() < 2:
        raise ValueError("target needs both classes")
    model = LogisticRegression(max_iter=1000).fit(d[features], d[target].astype(int))
    return model, features


def beta_smoothed_rate(
    events: float, trials: float, alpha: float = 1.0, beta: float = 9.0
) -> float:
    return float((events + alpha) / (trials + alpha + beta))


def reliability_features(
    df: pd.DataFrame, group_col="car", dnf_col="dnf", window: int = 10
) -> pd.Series:
    d = df.copy()
    d[dnf_col] = _num(d[dnf_col])

    def f(s):
        s = s.shift(1)
        return s.rolling(window, min_periods=1).apply(
            lambda x: beta_smoothed_rate(x.sum(), len(x)), raw=False
        )

    return d.groupby(group_col, sort=False)[dnf_col].transform(f)


def pit_loss(entry_time_s: float, exit_time_s: float, reference_green_time_s: float) -> float:
    return float(exit_time_s - entry_time_s - reference_green_time_s)


def undercut_power(
    pre_pit_pace: float, new_tyre_pace: float, pit_loss_s: float, laps_to_intersection: float = 2.0
) -> float:
    """Positive = stronger undercut: pace gain over next laps minus pit-loss burden scale."""
    gain = max(0.0, float(pre_pit_pace) - float(new_tyre_pace)) * float(laps_to_intersection)
    return gain / max(float(pit_loss_s), EPS)


def backward_asof_weather(
    telemetry: pd.DataFrame, weather: pd.DataFrame, time_col="timestamp"
) -> pd.DataFrame:
    t = telemetry.copy()
    w = weather.copy()
    t[time_col] = pd.to_datetime(t[time_col], utc=True)
    w[time_col] = pd.to_datetime(w[time_col], utc=True)
    return pd.merge_asof(
        t.sort_values(time_col), w.sort_values(time_col), on=time_col, direction="backward"
    )


def resample_telemetry(df: pd.DataFrame, distance_step_m: float = 5.0) -> pd.DataFrame:
    d = df.sort_values("distance").drop_duplicates("distance").copy()
    dist = _num(d.distance).to_numpy()
    grid = np.arange(np.nanmin(dist), np.nanmax(dist) + distance_step_m, distance_step_m)
    out = pd.DataFrame({"distance": grid})
    for col in d.columns:
        if col == "distance":
            continue
        vals = _num(d[col])
        if vals.notna().sum() >= 2:
            out[col] = np.interp(grid, dist[vals.notna()], vals[vals.notna()])
    return out


def robust_group_scale(df: pd.DataFrame, col: str, group="raceId") -> pd.Series:
    def scale(s):
        x = _num(s)
        med = x.median()
        mad = (x - med).abs().median()
        return (x - med) / (1.4826 * mad if mad > EPS else 1.0)

    return df.groupby(group)[col].transform(scale)
