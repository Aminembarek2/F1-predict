"""Evaluate candidates on tuning races; produce measured adoption decisions."""

import argparse
import json
import logging
import xml.etree.ElementTree as ET

import pandas as pd

from f1pred.artifacts import write_manifest
from f1pred.cli import _load_panel
from f1pred.config import RAW_DIR, RESULTS_DIR, SEED, ensure_dirs
from f1pred.evaluation.experiments import acceptance_report, run_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=10000)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    ensure_dirs()
    output = RESULTS_DIR / "experiments"
    output.mkdir(exist_ok=True)
    if args.report_only:
        predictions = pd.read_parquet(output / "predictions.parquet")
        metrics = pd.read_csv(output / "per_race.csv")
    else:
        panel = _load_panel()
        practice_path = RAW_DIR / "practice.parquet"
        practice = pd.read_parquet(practice_path) if practice_path.exists() else None
        predictions, metrics, calibration = run_candidates(
            panel, n_sims=args.sims, practice=practice
        )
        predictions.to_parquet(output / "predictions.parquet", index=False)
        metrics.to_csv(output / "per_race.csv", index=False)
        calibration.to_csv(output / "calibration_folds.csv", index=False)
        write_manifest(
            output / "manifest.json",
            command="python -m scripts.run_experiments",
            n_sims=args.sims,
            seed=SEED,
            tuning_seasons=[2022, 2023, 2024, 2025],
            warmup_seasons=[2021],
            practice_available=practice is not None,
            evaluation_scope="retrospective_conditional_field",
        )
    junit = RESULTS_DIR / "tests.xml"
    tests_passed = False
    if junit.exists():
        suites = ET.parse(junit).getroot().findall("testsuite")
        tests_passed = bool(suites) and all(
            int(s.attrib["tests"]) > 0
            and int(s.attrib.get("failures", 0)) == 0
            and int(s.attrib.get("errors", 0)) == 0
            for s in suites
        )
    report = acceptance_report(predictions, metrics, tests_passed=tests_passed)
    if "corrected_practice" not in set(predictions["model"]):
        report["practice_status"] = (
            "not_evaluable: no eligible practice corpus with enough earlier calibration races"
        )
    report["winner_ece_bin_edges"] = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.55, 1.0]
    clean = json.loads(pd.Series([report]).to_json(orient="values"))[0]
    (output / "acceptance.json").write_text(json.dumps(clean, indent=2) + "\n")
    pd.DataFrame(report["leaderboard"]).to_csv(output / "leaderboard.csv", index=False)
    print(pd.DataFrame(report["leaderboard"]).round(4).to_string(index=False))
    print("Adopted:", [r["model"] for r in report["decisions"] if r["adopt"]])


if __name__ == "__main__":
    main()
