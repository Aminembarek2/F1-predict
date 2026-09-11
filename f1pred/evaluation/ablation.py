"""Feature-family ablation.

Each module in the architecture is switched off in turn and the whole pipeline is
re-run over the tuning window. A component is kept only when removing it makes
the forecast measurably worse; "it is physically motivated" is not evidence.

Ablations run on the **tuning** window (2022-2025). Running them on the held-out
season would turn the holdout into a selection set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import DEFAULT_MODEL_CONFIG, ModelConfig, regime_for
from ..features.pace import entry_strength, fit_strength
from ..features.reliability import fit_reliability
from ..pipeline import Panel, resolve_circuit
from ..sim.race import SimulationSettings, simulate_race
from .metrics import evaluate, paired_bootstrap, summarise

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AblationVariant:
    """One configuration of the pipeline."""

    name: str
    description: str
    use_simulator: bool = True
    use_reliability: bool = True
    use_circuit: bool = True
    use_qualifying_obs: bool = True
    #: DNF probability applied to everyone when the reliability model is off.
    flat_dnf: float = 0.12
    #: Overtaking difficulty applied everywhere when the circuit model is off.
    flat_overtaking: float = 0.5


DEFAULT_VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant(
        "strength_only", "rank directly by latent strength; no race simulation", use_simulator=False
    ),
    AblationVariant(
        "no_qualifying",
        "strength fitted from race results only, no qualifying observations",
        use_qualifying_obs=False,
    ),
    AblationVariant(
        "no_reliability", "simulator with one flat DNF rate for every entry", use_reliability=False
    ),
    AblationVariant(
        "no_circuit",
        "simulator with one flat overtaking difficulty for every venue",
        use_circuit=False,
    ),
    AblationVariant("full", "every component enabled"),
)


def run_ablation(
    panel: Panel,
    race_ids: list[str],
    entries_by_race: dict[str, pd.DataFrame],
    *,
    variants: tuple[AblationVariant, ...] = DEFAULT_VARIANTS,
    config: ModelConfig = DEFAULT_MODEL_CONFIG,
    n_sims: int = 8_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every variant over the same races and return predictions and metrics."""
    settings = SimulationSettings(n_sims=n_sims, seed=config.seed)
    contexts = {
        rid: resolve_circuit(panel, rid, str(entries_by_race[rid]["circuit_id"].iloc[0]))
        for rid in race_ids
    }
    observations_all = panel.observations
    observations_race_only = observations_all.loc[observations_all["kind"] == "race"]

    frames: list[pd.DataFrame] = []
    for variant in variants:
        log.info("ablation variant %s", variant.name)
        obs = observations_all if variant.use_qualifying_obs else observations_race_only
        for rid in race_ids:
            entries = entries_by_race[rid]
            idx = panel.index_of(rid)
            regime = regime_for(int(entries["season"].iloc[0]))
            est = fit_strength(obs, as_of_index=idx, target_regime=regime, config=config)
            scored = entry_strength(est, entries)

            if not variant.use_simulator:
                row = scored.copy()
                row["score"] = row["strength"]
                row["win_probability"] = np.nan
                row["model"] = variant.name
                row["race_id"] = rid
                frames.append(row)
                continue

            ctx = contexts[rid]
            if variant.use_reliability:
                rel = fit_reliability(
                    panel.results, as_of_index=idx, target_regime=regime, config=config
                )
                multiplier = ctx.dnf_multiplier if variant.use_circuit else 1.0
                scored["dnf_probability"] = [
                    rel.hazard(d, c, multiplier=multiplier)
                    for d, c in zip(scored["driver_id"], scored["constructor_id"], strict=True)
                ]
            else:
                scored["dnf_probability"] = variant.flat_dnf

            weight = scored["driver_obs_weight"].to_numpy(float)
            scored["strength_sd"] = np.where(scored["is_rookie"], 0.45, 0.45 / (1.0 + weight / 3.0))
            difficulty = (
                ctx.overtaking_difficulty if variant.use_circuit else variant.flat_overtaking
            )
            table = simulate_race(scored, overtaking_difficulty=difficulty, settings=settings).table
            table["score"] = table["pace_rank_score"]
            table["model"] = variant.name
            table["race_id"] = rid
            frames.append(table)

    predictions = pd.concat(frames, ignore_index=True, sort=False)
    per_race = []
    for model, g in predictions.groupby("model"):
        m = evaluate(g, score_col="score", prob_col="win_probability")
        m["model"] = model
        per_race.append(m)
    return predictions, pd.concat(per_race, ignore_index=True)


def ablation_report(per_race: pd.DataFrame, *, reference: str = "full") -> pd.DataFrame:
    """Summarise each variant and bootstrap its difference against the full model."""
    tables = {str(m): g for m, g in per_race.groupby("model")}
    rows = []
    for name, table in tables.items():
        summary = summarise(table, name)
        if name != reference and reference in tables:
            for metric in ("top1", "ndcg5", "spearman", "rank_mae"):
                test = paired_bootstrap(table, tables[reference], metric)
                summary[f"{metric}_delta_vs_full"] = test["delta"]
                summary[f"{metric}_ci_low"] = test["ci_low"]
                summary[f"{metric}_ci_high"] = test["ci_high"]
        rows.append(summary)
    return pd.DataFrame(rows).sort_values("top1", ascending=False).reset_index(drop=True)
