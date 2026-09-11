"""Chronological test of hierarchical dynamic pace, confined to 2022–2025.

Annual variance fits use only earlier years. States update after each completed
race. 2021 supplies out-of-fold calibration warm-up. No 2026 outcomes are read by
any fit or used to choose settings. Fields are retrospective, so these results
cannot alone authorize adoption. Run pytest, then --report-only to refresh gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from f1pred.artifacts import write_manifest
from f1pred.cli import _load_panel
from f1pred.config import RESULTS_DIR, SEED, regime_for
from f1pred.evaluation.backtest import leaderboard, race_entries, walk_forward
from f1pred.evaluation.experiments import acceptance_report, winner_only
from f1pred.evaluation.metrics import evaluate, probability_report
from f1pred.evaluation.uncertainty import paired_block_bootstrap
from f1pred.features.pace import normal_scores
from f1pred.models.calibration import expected_calibration_error, reliability_bins
from f1pred.models.ranking import fit_rank_calibration, rank_probabilities
from f1pred.models.retirement import fit_retirement
from f1pred.models.state_space import (
    filtered_state,
    fit_variances,
    sample_ranking,
    timing_observations,
)

log = logging.getLogger(__name__)


def report(out: Path) -> None:
    predictions = pd.read_parquet(out / "predictions.parquet")
    metrics = pd.read_csv(out / "per_race.csv")
    junit = RESULTS_DIR / "tests.xml"
    tests_passed = False
    if junit.exists():
        suites = list(ET.parse(junit).getroot().iter("testsuite"))
        tests_passed = (
            bool(suites)
            and sum(int(s.get("tests", 0)) for s in suites) > 0
            and all(int(s.get("failures", 0)) == int(s.get("errors", 0)) == 0 for s in suites)
        )
    acceptance = acceptance_report(predictions, metrics, tests_passed=tests_passed)
    fits = pd.read_json(out / "variance_folds.json")
    fit_ok = bool(fits.converged.all()) and bool(
        (fits.last_training_index < fits.as_of_index).all()
    )
    chronology = predictions[["race_id", "season", "event_index"]].drop_duplicates()
    block_rows = []
    for decision in acceptance["decisions"]:
        a = metrics.loc[metrics.model.eq(decision["incumbent"])]
        b = metrics.loc[metrics.model.eq(decision["model"])]
        block_comparisons = {}
        for metric in ("top1", "ndcg5", "winner_logloss"):
            comparisons = [
                paired_block_bootstrap(a, b, metric, chronology, block_size=size) for size in (4, 8)
            ]
            block_comparisons[metric] = comparisons
            block_rows.extend(
                {"model": decision["model"], "metric": metric, **c} for c in comparisons
            )
        decision["block_sensitivity"] = block_comparisons
        decision["gates"]["block_sensitivity_supports_improvement"] = any(
            all(
                c["p_better"] >= 0.975
                and (c["ci_high"] < 0 if m == "winner_logloss" else c["ci_low"] > 0)
                for c in comparisons
            )
            for m, comparisons in block_comparisons.items()
        )
        decision["gates"]["chronological_variance_fits_converged"] = fit_ok
        decision["adopt"] = all(decision["gates"].values())
    pd.DataFrame(block_rows).to_csv(out / "block_bootstrap.csv", index=False)
    probability_rows, winner_bins = [], []
    for name, group in predictions.groupby("model"):
        outcome_report = probability_report(group)
        if outcome_report.empty:
            continue
        outcome_report["model"] = name
        outcome_report["ece_binning"] = "10 equal-width bins"
        winner = outcome_report.outcome.eq("winner")
        if winner.any():
            outcome_report.loc[winner, "ece"] = expected_calibration_error(
                group, prob_col="win_probability", outcome_col="winner"
            )
            outcome_report.loc[winner, "ece_binning"] = "acceptance winner bins"
            winner_bins.append(
                reliability_bins(group, prob_col="win_probability", outcome_col="winner").assign(
                    model=name
                )
            )
        probability_rows.append(outcome_report)
    pd.concat(probability_rows, ignore_index=True).to_csv(
        out / "probability_report.csv", index=False
    )
    pd.concat(winner_bins, ignore_index=True).to_csv(
        out / "winner_calibration_bins.csv", index=False
    )
    acceptance["winner_ece_bin_edges"] = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.55, 1.0]
    acceptance["other_outcome_ece_bins"] = "10 equal-width bins"

    acceptance["test_evidence"] = (
        str(junit.relative_to(RESULTS_DIR.parent)) if junit.exists() else None
    )
    acceptance["test_report_sha256"] = (
        hashlib.sha256(junit.read_bytes()).hexdigest() if junit.exists() else None
    )
    acceptance["test_count"] = sum(int(s.get("tests", 0)) for s in suites) if junit.exists() else 0
    acceptance["hyperparameter_uncertainty"] = "conditional empirical Bayes; not integrated"
    acceptance["regulation_policy"] = (
        "restart constructor states at a regime boundary; carry driver contrasts"
    )
    # Native JSON does not accept NaN. Missing probabilities remain explicit nulls.
    clean = json.loads(pd.Series([acceptance]).to_json(orient="values"))[0]
    (out / "acceptance.json").write_text(json.dumps(clean, indent=2) + "\n")
    rows = [
        {"model": d["model"], "metric": metric, **comparison}
        for d in acceptance["decisions"]
        for metric, comparison in d["paired"].items()
    ]
    pd.DataFrame(rows).to_csv(out / "paired_tests.csv", index=False)
    write_manifest(
        out / "report_manifest.json",
        command="python -m scripts.run_state_space_experiment --report-only",
        predictions_sha256=hashlib.sha256((out / "predictions.parquet").read_bytes()).hexdigest(),
        test_report_sha256=acceptance["test_report_sha256"],
        winner_ece_bin_edges=acceptance["winner_ece_bin_edges"],
    )
    print(leaderboard(metrics).round(4).to_string(index=False))
    print("Adopted:", [d["model"] for d in acceptance["decisions"] if d["adopt"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=20000)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = RESULTS_DIR / "state_space_experiment"
    out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report(out)
        return
    if args.sims <= 0:
        parser.error("--sims must be positive")
    panel = _load_panel()
    # Remove held-out data before constructing measurement matrices or rosters.
    results = panel.results.loc[panel.results.season <= 2025].copy()
    qualifying = panel.qualifying.loc[panel.qualifying.season <= 2025].copy()
    observations = timing_observations(results, qualifying)
    ids = (
        results.loc[results.season.between(2021, 2025)]
        .sort_values("event_index")
        .race_id.unique()
        .tolist()
    )
    coverage = []
    for o in observations:
        coverage.append(
            {
                "race_id": o.race_id,
                "season": o.season,
                "kind": o.kind,
                "timed_entries": len(o.entries),
                "roster_entries": len(o.roster),
            }
        )
    pd.DataFrame(coverage).to_csv(out / "timing_coverage.csv", index=False)
    base, _ = walk_forward(
        panel,
        ids,
        n_sims=args.sims,
        baselines=("championship", "last_race", "form5", "uniform"),
        allow_retrospective=True,
    )
    base["event_index"] = base.race_id.map(panel.event_order.set_index("race_id").event_index)
    frames, folds, retirement_folds = [], [], []
    timing_checks = []
    state, fit, fit_year = None, None, None
    for rid in ids:
        entries = race_entries(panel, rid)
        idx, year = panel.index_of(rid), int(entries.season.iloc[0])
        if fit is None or year != fit_year:
            log.info("fitting variance components before %s", rid)
            fit = fit_variances(observations, as_of_index=idx)
            fit_year = year
            folds.append({"season": year, **asdict(fit)})
            (out / "variance_folds.json").write_text(json.dumps(folds, indent=2) + "\n")
            state = filtered_state(observations, fit, as_of_index=idx)
            log.info("variance fit: %s; boundary=%s", asdict(fit.variances), fit.at_boundary)
        mean, latent, predictive = state.predict(entries, event_index=idx, season=year)
        retirement = fit_retirement(results, as_of_index=idx, target_regime=regime_for(year))
        retirement_folds.append({"race_id": rid, "as_of_index": idx, **asdict(retirement)})
        row = sample_ranking(entries, mean, predictive, n_sims=args.sims, retirement=retirement)
        row["model"] = "state_space"
        row["event_index"] = idx
        row["stage"] = "PRE_WEEKEND"
        row["evaluation_scope"] = "retrospective_conditional_field"
        row["latent_contrast_sd"] = row.driver_id.map(
            dict(zip(entries.driver_id, np.sqrt(np.maximum(np.diag(latent), 0)), strict=True))
        )
        row["race_predictive_sd"] = row.driver_id.map(
            dict(zip(entries.driver_id, np.sqrt(np.maximum(np.diag(predictive), 0)), strict=True))
        )
        frames.append(row)
        # Evaluate observed timing contrasts against the pre-weekend posterior,
        # before either session updates it. Pairwise rows are dependent; report
        # coverage per race rather than pretending every pair is a new label.
        entry_index = {d: i for i, d in enumerate(entries.driver_id)}
        for observation in observations:
            if observation.race_id != rid or year < 2022:
                continue
            keep = [i for i, d in enumerate(observation.entries.driver_id) if d in entry_index]
            locations = [entry_index[observation.entries.driver_id.iloc[i]] for i in keep]
            if len(keep) < 2:
                continue
            error = observation.values[keep] - mean[locations]
            covariance = latent[np.ix_(locations, locations)] + np.eye(len(keep)) * getattr(
                fit.variances, observation.kind
            )
            left, right = np.triu_indices(len(keep), k=1)
            variance = (
                covariance[left, left] + covariance[right, right] - 2 * covariance[left, right]
            )
            z = (error[left] - error[right]) / np.sqrt(variance)
            timing_checks.append(
                {
                    "race_id": rid,
                    "season": year,
                    "kind": observation.kind,
                    "timed_entries": len(keep),
                    "n_pairs": len(z),
                    "mean_squared_standardized_error": float(np.mean(z * z)),
                    "contrast_interval_80_coverage": float(np.mean(abs(z) <= 1.281551566)),
                    "contrast_interval_95_coverage": float(np.mean(abs(z) <= 1.959963985)),
                }
            )
        # Only now can this race update states for the next forecast.
        for o in observations:
            if o.race_id == rid:
                state.observe(o)
        log.info("predicted %s using history < %.0f", rid, idx)
    history = pd.concat([base, *frames], ignore_index=True)
    calibrated, calibration_folds = [], []
    for source in ("state_space", "championship"):
        training = history.loc[history.model.eq(source)].copy()
        if source == "championship":
            rank = training.groupby("race_id").score.rank(ascending=False, method="average")
            training["calibration_score"] = normal_scores(
                rank, training.groupby("race_id").driver_id.transform("size")
            )
        else:
            training["calibration_score"] = training.strength
        for rid, group in training.groupby("race_id", sort=True):
            if group.season.iloc[0] < 2022:
                continue
            calibration = fit_rank_calibration(
                training,
                as_of_index=float(group.event_index.iloc[0]),
                score_col="calibration_score",
            )
            group = group.copy()
            group["win_probability"] = rank_probabilities(
                group.calibration_score, calibration.temperature
            )
            group["model"] = f"calibrated_{source}"
            calibrated.append(winner_only(group))
            calibration_folds.append(
                {"race_id": rid, "model": group.model.iloc[0], **asdict(calibration)}
            )
    predictions = pd.concat(
        [history.loc[history.season.between(2022, 2025)], *calibrated], ignore_index=True
    )
    metrics = pd.concat(
        [evaluate(group).assign(model=name) for name, group in predictions.groupby("model")],
        ignore_index=True,
    )
    predictions.to_parquet(out / "predictions.parquet", index=False)
    metrics.to_csv(out / "per_race.csv", index=False)
    leaderboard(metrics).to_csv(out / "leaderboard.csv", index=False)
    pd.DataFrame(timing_checks).to_csv(out / "timing_validation.csv", index=False)
    pd.DataFrame(calibration_folds).to_csv(out / "calibration_folds.csv", index=False)
    (out / "retirement_folds.json").write_text(json.dumps(retirement_folds, indent=2) + "\n")
    write_manifest(
        out / "manifest.json",
        command="python -m scripts.run_state_space_experiment",
        n_sims=args.sims,
        seed=SEED,
        tuning_seasons=[2022, 2023, 2024, 2025],
        warmup_season=2021,
        variance_refit="before each season, earlier races only",
        evaluation_scope="retrospective_conditional_field",
    )
    report(out)


if __name__ == "__main__":
    main()
