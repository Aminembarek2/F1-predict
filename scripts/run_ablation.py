"""Switch each component off in turn and measure what its absence costs.

Ablations run on the tuning window only. Running them on the held-out season
would turn that season into a selection set, which is the mistake the whole
evaluation design exists to avoid.
"""

import argparse
import logging

import pandas as pd

from f1pred.artifacts import write_manifest
from f1pred.cli import _load_panel
from f1pred.config import RESULTS_DIR, SEED, ensure_dirs
from f1pred.evaluation.ablation import ablation_report, run_ablation
from f1pred.evaluation.backtest import race_entries

TUNING_SEASONS = (2022, 2023, 2024, 2025)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=8000)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(TUNING_SEASONS))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if set(args.seasons) - set(TUNING_SEASONS):
        raise SystemExit("ablations are restricted to the tuning window")
    ensure_dirs()

    panel = _load_panel()
    ids = (
        panel.results.loc[panel.results["season"].isin(args.seasons)]
        .sort_values("event_index")["race_id"]
        .unique()
        .tolist()
    )
    entries, race_ids = {}, []
    for race_id in ids:
        # Retrospective field membership, exactly as in the conditional backtest:
        # the ablation compares variants with each other, not with reality.
        truth = race_entries(panel, race_id)
        if truth.empty or truth["winner"].sum() != 1:
            continue
        entries[race_id] = truth
        race_ids.append(race_id)

    predictions, per_race = run_ablation(panel, race_ids, entries, n_sims=args.sims)
    report = ablation_report(per_race)

    output = RESULTS_DIR / "ablation"
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    per_race.to_csv(output / "per_race.csv", index=False)
    report.to_csv(output / "report.csv", index=False)
    write_manifest(
        output / "manifest.json",
        command="python -m scripts.run_ablation",
        seasons=args.seasons,
        n_races=len(race_ids),
        n_sims=args.sims,
        seed=SEED,
        evaluation_scope="retrospective_conditional_field",
    )
    with pd.option_context("display.width", 200):
        print(report.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
