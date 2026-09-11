"""Fit the simulator's variance budget on the tuning window.

The strength and reliability models are fitted once per race, walk-forward, and
cached; only the simulator is re-run inside the optimiser. That keeps the search
affordable without weakening the chronological rule, because every cached fit
already saw nothing but earlier races.
"""

import argparse
import json
import logging
from dataclasses import asdict

import pandas as pd

from f1pred.artifacts import write_manifest
from f1pred.cli import _load_panel
from f1pred.config import DEFAULT_MODEL_CONFIG, RESULTS_DIR, SEED, ensure_dirs, regime_for
from f1pred.evaluation.backtest import race_entries
from f1pred.features.pace import entry_strength, fit_strength
from f1pred.features.reliability import fit_reliability
from f1pred.pipeline import resolve_circuit
from f1pred.sim.calibrate import BOUNDS, fit

TUNING = (2022, 2023, 2024, 2025)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(TUNING))
    parser.add_argument("--sims", type=int, default=4000)
    parser.add_argument("--max-evaluations", type=int, default=120)
    parser.add_argument("--race-score", choices=["rank", "pace"], default="rank")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if set(args.seasons) - set(TUNING):
        raise SystemExit("the variance budget is fitted on the tuning window only")
    ensure_dirs()

    panel = _load_panel(race_score=args.race_score)
    ids = (
        panel.results.loc[panel.results["season"].isin(args.seasons)]
        .sort_values("event_index")["race_id"]
        .unique()
        .tolist()
    )

    races = []
    for race_id in ids:
        truth = race_entries(panel, race_id)
        if truth.empty or truth["winner"].sum() != 1:
            continue
        idx = panel.index_of(race_id)
        regime = regime_for(int(truth["season"].iloc[0]))
        strength = fit_strength(
            panel.observations, as_of_index=idx, target_regime=regime, config=DEFAULT_MODEL_CONFIG
        )
        reliability = fit_reliability(
            panel.results, as_of_index=idx, target_regime=regime, config=DEFAULT_MODEL_CONFIG
        )
        ctx = resolve_circuit(panel, race_id, str(truth["circuit_id"].iloc[0]))
        scored = entry_strength(strength, truth)
        scored["dnf_probability"] = [
            reliability.hazard(d, c, multiplier=ctx.dnf_multiplier)
            for d, c in zip(scored["driver_id"], scored["constructor_id"], strict=True)
        ]
        scored["strength_sd"] = pd.to_numeric(scored["strength_se"], errors="coerce").fillna(0.45)
        races.append((scored, ctx.overtaking_difficulty, truth["winner"].eq(1).to_numpy()))
        if len(races) % 20 == 0:
            logging.info("prepared %d races", len(races))

    logging.info("fitting on %d races", len(races))
    result = fit(races, n_sims=args.sims, max_iterations=args.max_evaluations)

    out = RESULTS_DIR / "simulator_calibration"
    out.mkdir(parents=True, exist_ok=True)
    result.trace.to_csv(out / "trace.csv", index=False)
    summary = {
        "race_score": args.race_score,
        "seasons": args.seasons,
        "n_races": result.n_races,
        "n_evaluations": result.n_evaluations,
        "n_sims_per_evaluation": args.sims,
        "incumbent": {name: getattr(result.incumbent, name) for name in BOUNDS},
        "fitted": {name: getattr(result.settings, name) for name in BOUNDS},
        "incumbent_winner_logloss": result.incumbent_logloss,
        "fitted_winner_logloss": result.fitted_logloss,
        "improvement": result.improvement,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_manifest(
        out / "manifest.json",
        command="python -m scripts.calibrate_simulator",
        seed=SEED,
        settings=asdict(result.settings),
        evaluation_scope="retrospective_conditional_field",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
