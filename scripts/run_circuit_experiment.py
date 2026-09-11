"""Does a measured circuit beat an inferred one?

The ablation says the circuit model contributes nothing to ranking. The
validation in FINDINGS §7b says the quantity it is fed is close to noise. This
decides between those two explanations by swapping the inferred index for a
counted pass rate and re-running the same races.

Counted passes start in 2023, so the comparison runs on the seasons where a
circuit's value can be built from earlier measured races.
"""

import argparse
import json
import logging

import pandas as pd

from f1pred.artifacts import write_manifest
from f1pred.cli import _load_panel
from f1pred.config import RESULTS_DIR, SEED, ensure_dirs
from f1pred.evaluation.backtest import leaderboard, walk_forward
from f1pred.evaluation.metrics import paired_bootstrap

METRICS = ("top1", "ndcg5", "spearman", "rank_mae", "winner_logloss")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--sims", type=int, default=20000)
    parser.add_argument("--stage", default="POST_QUALI")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    ensure_dirs()

    panel = _load_panel()
    if panel.overtakes.empty:
        raise SystemExit("no counted passes on disk; run scripts.ingest_race_events first")
    ids = (
        panel.results.loc[panel.results["season"].isin(args.seasons)]
        .sort_values("event_index")["race_id"]
        .unique()
        .tolist()
    )

    runs = {}
    for label, measured in (("inferred", False), ("measured", True)):
        logging.info("circuit source: %s", label)
        predictions, per_race = walk_forward(
            panel,
            ids,
            stage=args.stage,
            n_sims=args.sims,
            baselines=("championship",),
            allow_retrospective=True,
            measured_overtaking=measured,
        )
        runs[label] = per_race.loc[per_race["model"].eq("strength_mc")].copy()
        runs[label]["circuit_source"] = label

    per_race = pd.concat(runs.values(), ignore_index=True)
    board = leaderboard(per_race.assign(model=per_race["circuit_source"]))
    paired = pd.DataFrame(
        [{"metric": m, **paired_bootstrap(runs["inferred"], runs["measured"], m)} for m in METRICS]
    )

    out = RESULTS_DIR / f"circuit_experiment_{args.stage.lower()}"
    out.mkdir(parents=True, exist_ok=True)
    per_race.to_csv(out / "per_race.csv", index=False)
    board.to_csv(out / "leaderboard.csv", index=False)
    paired.to_csv(out / "paired_tests.csv", index=False)
    write_manifest(
        out / "manifest.json",
        command="python -m scripts.run_circuit_experiment",
        seasons=args.seasons,
        stage=args.stage,
        n_races=len(ids),
        n_sims=args.sims,
        seed=SEED,
        evaluation_scope="retrospective_conditional_field",
    )
    print(board.round(4).to_string(index=False))
    print()
    print(paired.round(4).to_string(index=False))
    verdict = {
        m: bool(
            row["p_better"] >= 0.975
            and (row["ci_high"] < 0 if m in {"rank_mae", "winner_logloss"} else row["ci_low"] > 0)
        )
        for m, row in paired.set_index("metric").iterrows()
    }
    print("\nimproved on:", [m for m, ok in verdict.items() if ok] or "nothing")
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")


if __name__ == "__main__":
    main()
