"""Offline evidence for the September 2026 code review.

Run from the repository root with:
    .venv/bin/python -m research.review_checks --rerun

This measures existing behavior, including defects; it does not mark those defects
as passing tests, change model settings, or overwrite the original backtests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from f1pred.config import ROOT, SEED
from f1pred.evaluation.backtest import leaderboard, race_entries, walk_forward
from f1pred.evaluation.metrics import evaluate
from f1pred.features.pace import fit_strength
from f1pred.forecast import expected_entry_list, forecast_upcoming
from f1pred.leakage import LeakageError, assert_observed_before, shifted_expanding
from f1pred.models.calibration import expected_calibration_error
from f1pred.pipeline import build_panel
from f1pred.sim.race import SimulationSettings, simulate_race


def maximum_difference(a: pd.DataFrame, b: pd.DataFrame, keys: list[str]) -> float:
    """Compare matching rows; refuse to conceal missing entries or columns."""
    left, right = a.set_index(keys).sort_index(), b.set_index(keys).sort_index()
    pd.testing.assert_index_equal(left.index, right.index)
    cols = left.select_dtypes("number").columns
    delta = (left[cols] - right[cols]).abs()
    if not left[cols].isna().equals(right[cols].isna()):
        raise AssertionError("missing values differ")
    return float(delta.max().max())


def infer_simulation_count(probabilities: pd.Series) -> int:
    """Find a compatible count for a rerun; exact reproduction must still be checked."""
    p = probabilities.to_numpy()
    for count in (1000, 2000, 5000, 8000, 10000, 20000, 30000, 50000, 100000):
        if np.allclose(p * count, np.round(p * count), atol=1e-8, rtol=0):
            return count
    raise ValueError("no compatible simulation count in review search")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true", help="also regenerate all three backtests")
    parser.add_argument("--output", type=Path, default=ROOT / "results/review_2026_09_11.json")
    args = parser.parse_args()
    raw = {name: pd.read_parquet(ROOT / f"data/raw/{name}.parquet")
           for name in ("results", "qualifying", "schedule")}
    panel = build_panel(raw["results"], raw["qualifying"], raw["schedule"])
    evidence: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dependencies": {name: version(name) for name in
                         ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "pytest")},
        "backtests": {},
    }
    paths = list((ROOT / "f1pred").rglob("*.py"))
    paths += list((ROOT / "data/raw").glob("*.parquet"))
    paths += list((ROOT / "results").glob("backtest_*_predictions.parquet"))
    paths += [ROOT / "data/reference/circuit_specs.csv", Path(__file__)]
    evidence["input_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }

    for tag, stage in (("tuning", "PRE_WEEKEND"), ("2026_holdout", "PRE_WEEKEND"),
                       ("2026_postquali", "POST_QUALI")):
        saved = pd.read_parquet(ROOT / f"results/backtest_{tag}_predictions.parquet")
        frames = []
        for name, group in saved.groupby("model"):
            frame = evaluate(group)
            frame["model"] = name
            frames.append(frame)
        recalculated = pd.concat(frames, ignore_index=True)
        original_metrics = pd.read_csv(ROOT / f"results/backtest_{tag}_per_race.csv")
        model = saved.loc[saved["model"].eq("strength_mc")].copy()
        model["uniform"] = 1.0 / model.groupby("race_id")["driver_id"].transform("size")
        count = infer_simulation_count(model["win_probability"])
        item = {
            "n_races": int(model["race_id"].nunique()),
            "compatible_simulation_count": count,
            "stored_metrics_max_abs_difference": maximum_difference(
                recalculated, original_metrics, ["race_id", "model"]),
            "leaderboard": leaderboard(recalculated).to_dict("records"),
            "win_ece": expected_calibration_error(
                model, prob_col="win_probability", outcome_col="winner"),
            "uniform_win_ece": expected_calibration_error(
                model, prob_col="uniform", outcome_col="winner"),
            "displayed_rank_top1": float(model.loc[model["predicted_rank"].eq(1), "winner"].mean()),
            "row_pooled_dnf_observed": float(model["dnf"].mean()),
            "row_pooled_dnf_displayed": float(model["dnf_probability"].mean()),
            "row_pooled_dnf_simulated": float(model["dnf_probability_sim"].mean()),
            "row_pooled_interval_coverage": float(
                model["position_order"].between(model["finish_p10"], model["finish_p90"]).mean()),
        }
        if args.rerun:
            ids = model.sort_values(["season", "round"])["race_id"].unique().tolist()
            print(f"Regenerating {tag}: {len(ids)} races, {count} simulations each", flush=True)
            fresh, _ = walk_forward(panel, ids, stage=stage, n_sims=count)
            item["rerun_predictions_max_abs_difference"] = maximum_difference(
                fresh, saved, ["race_id", "model", "driver_id"])
        evidence["backtests"][tag] = item
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")

    print("Running targeted defect probes", flush=True)
    observed_indices = panel.observations[["race_id", "event_index"]].drop_duplicates()
    indices = observed_indices.merge(panel.event_order[["race_id", "event_index"]],
                                     on="race_id", suffixes=("_obs", "_panel"))
    evidence["intact_data_index_mismatches"] = int(
        indices["event_index_obs"].ne(indices["event_index_panel"]).sum())

    # Drop an earlier event's observations while retaining its scheduled slot.
    # The target's observations must remain illegal despite this ingestion gap.
    missing_race, target = "2014_01", "2025_05"
    partial = build_panel(
        raw["results"].loc[raw["results"]["race_id"].ne(missing_race)],
        raw["qualifying"].loc[raw["qualifying"]["race_id"].ne(missing_race)], raw["schedule"])
    idx = partial.index_of(target)
    admitted = partial.observations.loc[partial.observations["event_index"].lt(idx)]
    contaminated = partial.observations.copy()
    target_mask = contaminated["race_id"].eq(target)
    contaminated.loc[target_mask, "score"] *= -1
    fits = [fit_strength(obs, as_of_index=idx, target_regime="GROUND_EFFECT_2022")
            for obs in (partial.observations, contaminated)]
    changed = maximum_difference(fits[0].drivers, fits[1].drivers, ["driver_id"])
    evidence["missing_event_index_leak"] = {
        "removed_event": missing_race, "target": target,
        "target_observations_admitted": int(admitted["race_id"].eq(target).sum()),
        "driver_effect_max_change_when_target_scores_corrupted": changed,
    }

    invalid = {}
    for label, value in (("missing", None), ("malformed", "not-a-timestamp")):
        try:
            assert_observed_before(pd.DataFrame({"observed": [value]}), "observed", "2026-09-11")
        except LeakageError:
            invalid[label] = "rejected"
        else:
            invalid[label] = "accepted"
    evidence["timestamp_guard_invalid_input"] = invalid
    duplicate_event = pd.DataFrame({"team": ["a", "a", "a"],
                                    "event": [1, 1, 2], "value": [10.0, 20.0, 30.0]})
    shifted = shifted_expanding(duplicate_event, ["team"], "value", "event")
    evidence["shifted_aggregation_same_event_value"] = float(shifted.iloc[1])

    entries = race_entries(panel, "2026_13")
    pre = forecast_upcoming(panel, season=2026, round_no=13, entries=entries, n_sims=2000)
    post = forecast_upcoming(panel, season=2026, round_no=13, entries=entries,
                             stage="POST_QUALI", n_sims=2000)
    evidence["live_postquali"] = {
        "grid_supplied_in_entries": "grid" in entries,
        "still_simulates_qualifying": "pole_probability" in post.table,
        "pre_vs_post_win_probability_max_difference": maximum_difference(
            pre.table[["driver_id", "win_probability"]],
            post.table[["driver_id", "win_probability"]], ["driver_id"]),
    }
    pit_entries = race_entries(panel, "2022_05")
    predictions, _ = walk_forward(panel, ["2022_05"], stage="POST_QUALI", n_sims=100)
    retained = set(predictions.loc[predictions["model"].eq("strength_mc"), "driver_id"])
    evidence["pitlane_entries_dropped"] = sorted(set(pit_entries["driver_id"]) - retained)
    expected = expected_entry_list(panel, 2025, 1)
    actual = race_entries(panel, "2025_01")
    evidence["season_opener_entry_mismatch"] = {
        "previous_round_entries_not_among_actual_starters": sorted(
            set(expected["driver_id"]) - set(actual["driver_id"])),
        "actual_starters_missing_from_live_entries": sorted(
            set(actual["driver_id"]) - set(expected["driver_id"])),
    }

    toy = pd.DataFrame({"driver_id": [f"d{i}" for i in range(20)],
                        "constructor_id": [f"t{i // 2}" for i in range(20)],
                        "strength": 0.0, "dnf_probability": 0.1})
    simulation = simulate_race(toy, overtaking_difficulty=0.5,
                               settings=SimulationSettings(n_sims=100000, seed=SEED))
    evidence["dnf_chaos_probe"] = {
        "input_probability": 0.1,
        "analytical_marginal_probability": 0.1 * (1 + 0.22 * (2.1 - 1)),
        "simulated_probability": float(simulation.table["dnf_probability_sim"].mean()),
    }
    madrid = forecast_upcoming(panel, season=2026, round_no=14)
    saved_madrid = pd.read_csv(ROOT / "results/forecast_2026_14_PRE_WEEKEND.csv")
    evidence["madrid_rerun_max_abs_difference"] = maximum_difference(
        madrid.table, saved_madrid, ["driver_id"])
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Evidence written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
