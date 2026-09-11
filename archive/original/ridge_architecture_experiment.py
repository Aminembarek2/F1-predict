"""Does the rebuilt architecture beat the one it replaces?

Four changes are on trial, and they are entangled by design:

1. race pace measured in lap time rather than rank-transformed finishing order;
2. strength uncertainty taken from the ridge posterior, with the team/driver
   covariance, instead of a chosen formula;
3. the simulator's variance budget fitted by maximum likelihood;
4. (1)-(3) together, which is the only combination that is internally coherent -
   the fitted budget was fitted against the measured uncertainty.

The variance budget was fitted on 2024-2025, so it is evaluated here on
**2022-2023**, which that fit never saw. Both seasons are inside the tuning
window, so no held-out season is spent on this.
"""

import argparse
import json
import logging
from dataclasses import replace

import pandas as pd

from f1pred.artifacts import write_manifest
from f1pred.cli import _load_panel
from f1pred.config import (
    DEFAULT_MODEL_CONFIG,
    FITTED_VARIANCE_BUDGET,
    RESULTS_DIR,
    SEED,
    ensure_dirs,
)
from f1pred.evaluation.backtest import leaderboard, walk_forward
from f1pred.evaluation.metrics import paired_bootstrap, probability_report
from f1pred.sim.race import SimulationSettings

METRICS = ("top1", "ndcg5", "spearman", "rank_mae", "winner_logloss", "winner_brier")

#: Fitting seasons for the variance budget, excluded from this evaluation.
FITTED_ON = (2024, 2025)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023])
    parser.add_argument("--sims", type=int, default=20000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if set(args.seasons) & set(FITTED_ON):
        raise SystemExit(
            "the variance budget was fitted on 2024-2025; evaluating on those "
            "seasons would report an in-sample result as an improvement"
        )
    ensure_dirs()

    incumbent = SimulationSettings(n_sims=args.sims, seed=SEED)
    fitted = replace(incumbent, **FITTED_VARIANCE_BUDGET)

    heuristic = replace(DEFAULT_MODEL_CONFIG, strength_spread="heuristic")
    posterior = replace(DEFAULT_MODEL_CONFIG, strength_spread="posterior")
    inherited = replace(
        incumbent,
        qualifying_noise=0.55,
        race_pace_noise=0.60,
        start_swing_slots=1.6,
        strategy_swing_slots=2.0,
    )
    # Every factor is varied against the same control, so no change rides in on
    # another's result the way the first pass let the posterior spread do.
    arms = {
        "incumbent": ("rank", inherited, heuristic),
        "posterior_spread": ("rank", inherited, posterior),
        "fitted_budget": ("rank", fitted, heuristic),
        "pace_target": ("pace", inherited, heuristic),
        "rebuilt": ("pace", fitted, posterior),
    }

    per_race, calibration = [], []
    for name, (race_score, settings, model_config) in arms.items():
        logging.info("arm %s: race_score=%s", name, race_score)
        panel = _load_panel(race_score=race_score)
        ids = (
            panel.results.loc[panel.results["season"].isin(args.seasons)]
            .sort_values("event_index")["race_id"]
            .unique()
            .tolist()
        )
        predictions, metrics = walk_forward(
            panel,
            ids,
            baselines=("championship",),
            allow_retrospective=True,
            settings=settings,
            config=model_config,
        )
        model = metrics.loc[metrics["model"].eq("strength_mc")].copy()
        model["model"] = name
        per_race.append(model)
        report = probability_report(predictions.loc[predictions["model"].eq("strength_mc")])
        report["arm"] = name
        calibration.append(report)
        if name == "incumbent":
            baseline = metrics.loc[metrics["model"].eq("championship")].copy()
            baseline["model"] = "championship"
            per_race.append(baseline)

    per_race = pd.concat(per_race, ignore_index=True)
    calibration = pd.concat(calibration, ignore_index=True)
    board = leaderboard(per_race)

    control = per_race.loc[per_race["model"].eq("incumbent")]
    paired = pd.DataFrame(
        [
            {"arm": arm, "metric": m, **paired_bootstrap(control, group, m)}
            for arm, group in per_race.groupby("model")
            if arm not in {"incumbent", "championship"}
            for m in METRICS
        ]
    )

    out = RESULTS_DIR / "architecture_experiment"
    out.mkdir(parents=True, exist_ok=True)
    per_race.to_csv(out / "per_race.csv", index=False)
    board.to_csv(out / "leaderboard.csv", index=False)
    paired.to_csv(out / "paired_tests.csv", index=False)
    calibration.to_csv(out / "calibration.csv", index=False)
    write_manifest(
        out / "manifest.json",
        command="python -m scripts.run_architecture_experiment",
        seasons=args.seasons,
        fitted_budget_fitted_on=list(FITTED_ON),
        n_sims=args.sims,
        seed=SEED,
        evaluation_scope="retrospective_conditional_field",
    )

    with pd.option_context("display.width", 220):
        print(board.round(4).to_string(index=False))
        print()
        print(paired.round(4).to_string(index=False))
        print()
        print(calibration.round(4).to_string(index=False))

    adopted = [
        arm
        for arm, group in paired.groupby("arm")
        if any(
            row["p_better"] >= 0.975
            and (
                row["ci_high"] < 0
                if row["metric"] in {"rank_mae", "winner_logloss", "winner_brier"}
                else row["ci_low"] > 0
            )
            for _, row in group.iterrows()
        )
    ]
    (out / "verdict.json").write_text(json.dumps({"adopted": adopted}, indent=2) + "\n")
    print("\nClears the gate:", adopted or "nothing")


if __name__ == "__main__":
    raise SystemExit("Archived nonchronological experiment. Run python -m scripts.run_architecture_experiment instead.")
