"""Reproducible candidate comparisons; held-out seasons never select a model."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from ..features.pace import normal_scores
from ..features.practice import corrected_practice_pace
from ..models.calibration import expected_calibration_error
from ..models.ranking import fit_rank_calibration, rank_probabilities, simulate_plackett_luce
from ..pipeline import Panel
from .backtest import leaderboard, walk_forward
from .metrics import evaluate, paired_bootstrap


def winner_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Avoid carrying the incumbent's other probabilities into a new win model."""
    frame = frame.copy()
    for col in (
        "podium_probability",
        "top10_probability",
        "dnf_probability_sim",
        "dnf_probability",
        "expected_finish",
        "expected_finish_if_classified",
        "pole_probability",
        "finish_p10",
        "finish_p50",
        "finish_p90",
        "pace_rank_score",
    ):
        if col in frame:
            frame[col] = np.nan
    frame["probability_scope"] = "winner_only"
    frame = frame.sort_values(["score", "driver_id"], ascending=[False, True]).reset_index(
        drop=True
    )
    frame["predicted_rank"] = np.arange(1, len(frame) + 1)
    frame["win_probability_rank"] = (
        frame["win_probability"].rank(ascending=False, method="min").astype(int)
    )
    return frame


def run_candidates(
    panel: Panel, *, n_sims: int = 10000, practice: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """2021 warm-up followed by 2022–2025 chronological candidate evaluation.

    The repository's existing results permit only conditional-field diagnostics.
    Every output retains that limitation, and the adoption gate rejects it.
    """
    ids = (
        panel.results.loc[panel.results["season"].between(2021, 2025)]
        .sort_values("event_index")["race_id"]
        .unique()
        .tolist()
    )
    base_names = (
        ("championship", "last_race", "form5", "uniform")
        if not panel.standings.empty
        else ("gp_points", "last_race", "form5", "uniform")
    )
    base, _ = walk_forward(
        panel, ids, baselines=base_names, n_sims=n_sims, allow_retrospective=True
    )
    base["event_index"] = base["race_id"].map(panel.event_order.set_index("race_id")["event_index"])
    strength = base.loc[base["model"].eq("strength_mc")].copy()
    candidates, calibration_rows = [], []
    families = [(name, name) for name in base_names if name != "uniform"] + [
        ("latent_strength", "strength_mc")
    ]
    for name, source in families:
        history = base.loc[base["model"].eq(source)].copy()
        if name == "latent_strength":
            history["score"] = history["strength"]
        else:
            positions = history.groupby("race_id")["score"].rank(ascending=False, method="average")
            history["score"] = normal_scores(
                positions, history.groupby("race_id")["driver_id"].transform("size")
            )
        for rid, group in history.groupby("race_id", sort=True):
            idx = float(group["event_index"].iloc[0])
            fit = fit_rank_calibration(history, as_of_index=idx)
            if int(group["season"].iloc[0]) < 2022:
                continue
            row = group.copy()
            row["win_probability"] = rank_probabilities(row["score"], fit.temperature)
            row["model"] = f"calibrated_{name}"
            candidates.append(winner_only(row))
            calibration_rows.append({"race_id": rid, "model": row["model"].iloc[0], **asdict(fit)})

    pl_history = strength.assign(score=strength["strength"])
    for rid, group in strength.groupby("race_id", sort=True):
        if int(group["season"].iloc[0]) < 2022:
            continue
        fit = fit_rank_calibration(
            pl_history, as_of_index=float(group["event_index"].iloc[0]), objective="ranking"
        )
        row = simulate_plackett_luce(group, temperature=fit.temperature, n_sims=n_sims)
        row["model"] = "plackett_luce"
        candidates.append(row)
        calibration_rows.append({"race_id": rid, "model": "plackett_luce", **asdict(fit)})

    if practice is not None and not practice.empty:
        practice_frames = []
        for rid, laps in practice.groupby("race_id"):
            incumbent = strength.loc[strength["race_id"].eq(rid)].copy()
            if incumbent.empty:
                continue
            cutoff = pd.Timestamp(laps["cutoff_utc"].iloc[0])
            laps = laps.loc[pd.to_datetime(laps["observed_at"], utc=True) < cutoff]
            pace = corrected_practice_pace(laps, cutoff=cutoff)
            if len(pace) < 4:
                continue
            row = incumbent.merge(pace, on="driver_id", how="left", validate="one_to_one")
            # A fixed, declared blend is evaluated; no search uses 2026 outcomes.
            fraction = (0.35 / (1 + row["pace_sd"].fillna(np.inf))).clip(0, 0.35)
            row["score"] = (1 - fraction) * row["strength"] + fraction * row[
                "practice_score"
            ].fillna(0)
            row["stage"] = "POST_FP2"
            row["practice_used"] = row["practice_score"].notna()
            row["cutoff_utc"] = cutoff.isoformat()
            practice_frames.append(row)
        if practice_frames:
            history = pd.concat(practice_frames, ignore_index=True)
            for rid, group in history.groupby("race_id", sort=True):
                fit = fit_rank_calibration(history, as_of_index=float(group["event_index"].iloc[0]))
                # No in-sample fallback is presented as a calibrated result.
                if fit.n_races < 10:
                    continue
                for name in ("corrected_practice", "practice_control"):
                    row = group.copy()
                    if name == "practice_control":
                        row["score"] = row["strength"]
                        fit = fit_rank_calibration(
                            history.assign(score=history["strength"]),
                            as_of_index=float(group["event_index"].iloc[0]),
                        )
                        row["practice_used"] = False
                    row["win_probability"] = rank_probabilities(row["score"], fit.temperature)
                    row["model"] = name
                    candidates.append(winner_only(row))
                    calibration_rows.append({"race_id": rid, "model": name, **asdict(fit)})
    predictions = pd.concat(
        [base.loc[base["season"].between(2022, 2025)], *candidates], ignore_index=True
    )
    metrics = []
    for name, group in predictions.groupby("model"):
        per_race = evaluate(group)
        per_race["model"] = name
        metrics.append(per_race)
    return predictions, pd.concat(metrics, ignore_index=True), pd.DataFrame(calibration_rows)


def acceptance_report(
    predictions: pd.DataFrame, per_race: pd.DataFrame, *, tests_passed: bool
) -> dict:
    """Apply explicit tuning, calibration, bootstrap and provenance gates.

    No approval is inferred from a point estimate or a synthetic unit test.
    Retrospective field diagnostics can motivate a next experiment, not adoption.
    """
    if not predictions["season"].between(2022, 2025).all():
        raise ValueError("acceptance data must be exclusively in the tuning window")
    decisions = []
    primary = ("top1", "ndcg5", "winner_logloss")
    for model, group in predictions.groupby("model"):
        if (
            model in {"strength_mc", "uniform", "practice_control"}
            or group["win_probability"].isna().all()
        ):
            continue
        ids = set(group["race_id"])
        control_name = "practice_control" if model == "corrected_practice" else "strength_mc"
        control = predictions.loc[
            predictions["model"].eq(control_name) & predictions["race_id"].isin(ids)
        ]
        a = per_race.loc[per_race["model"].eq(control_name) & per_race["race_id"].isin(ids)]
        b = per_race.loc[per_race["model"].eq(model)]
        paired = {metric: paired_bootstrap(a, b, metric) for metric in primary}
        ece = expected_calibration_error(group, prob_col="win_probability", outcome_col="winner")
        control_ece = expected_calibration_error(
            control, prob_col="win_probability", outcome_col="winner"
        )
        improved = any(
            paired[m]["p_better"] >= 0.975
            and (paired[m]["ci_high"] < 0 if m == "winner_logloss" else paired[m]["ci_low"] > 0)
            for m in primary
        )
        gates = {
            "tests_passed": bool(tests_passed),
            "tuning_only": True,
            "enough_paired_races": len(ids) >= 30,
            "timestamped_entry_and_history_snapshots": bool(
                group["evaluation_scope"].eq("timestamped_snapshot").all()
            ),
            "calibration_not_degraded": bool(ece <= control_ece + 1e-12),
            "winner_brier_not_degraded": bool(
                b["winner_brier"].mean() <= a["winner_brier"].mean() + 1e-12
            ),
            "paired_primary_improvement": bool(improved),
        }
        decisions.append(
            {
                "model": model,
                "incumbent": control_name,
                "adopt": all(gates.values()),
                "gates": gates,
                "n_races": len(ids),
                "win_ece": ece,
                "incumbent_win_ece": control_ece,
                "paired": paired,
            }
        )
    return {
        "protocol": "chronological calibration; 2021 warm-up; 2022–2025 tuning",
        "primary_metrics": list(primary),
        "bootstrap_draws": 10000,
        "favourable_mass_required": 0.975,
        "decisions": decisions,
        "leaderboard": leaderboard(per_race).to_dict("records"),
    }
