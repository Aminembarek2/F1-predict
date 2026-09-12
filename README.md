# F1 Predictor

**Calibrated pre-race Formula 1 probabilities — and every rejected experiment on the record.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![tests 177 passing](https://img.shields.io/badge/tests-177%20passing-2ea043)](tests/)
[![ruff](https://img.shields.io/badge/lint-ruff-261230?logo=ruff&logoColor=white)](pyproject.toml)
[![licence MIT](https://img.shields.io/badge/licence-MIT-blue)](LICENSE)

A Monte Carlo race simulator that publishes win, podium, top-ten and retirement
probabilities before a car turns a wheel — wrapped in an evaluation harness built
to catch its own author cheating.

**It does not beat reading the championship table.** That is measured, not modest:
over 87 strictly scored races the two are level at 43 correct winners each. What
the system has that a standings table does not is *calibrated probability*, and
the evidence for both claims is in this repository.

![How the forecast is built: sources, one chronological panel, leakage gates, three fitted models, the Monte Carlo simulator, and the published probabilities](images/architecture.svg)

---

## Results at a glance

| | Model | Championship baseline |
|---|--:|--:|
| Winner accuracy, 87 strictly scored races (2022–25) | **49.4%** | 49.4% |
| Winner accuracy, held-out 2026 (12 races) | 16.7% | **41.7%** |
| Winner log-loss | **1.431** | — *(baselines emit no probabilities)* |
| Winner calibration error (ECE) | **0.008** | — |
| Finishes inside the published P10–P90 band | **79.8%** | — |

Read the middle row as written: on the one season the model had never seen, it
picked 2 winners in 12 against the baseline's 5. The bottom three rows are the
case for the project — no ranking baseline produces a probability at all, so
there is nothing to compare them against except a uniform 1/20.

---

## The forecast: Madrid 2026

Round 14, from an entry list dated 2026-09-07 and an information cutoff of
**2026-09-11T01:00Z** — 01:00 UTC on Friday, sixty hours before lights out. The
model used nothing after that cutoff, so no lap of this weekend informs it. The
artefact itself was written later that day, after Friday practice had run.
100,000 simulated races, seed `20260913`.

| # | Driver | Win | Podium | Top 10 | DNF risk |
|--:|---|--:|--:|--:|--:|
| 1 | Antonelli | **27.7%** | 62.3% | 93.4% | 6.3% |
| 2 | Russell | 19.7% | 51.1% | 86.1% | 13.6% |
| 3 | Norris | 14.9% | 44.0% | 85.0% | 14.6% |
| 4 | Hamilton | 13.3% | 43.2% | 92.4% | 6.8% |
| 5 | Leclerc | 9.3% | 33.1% | 82.1% | 16.9% |
| 6 | Piastri | 7.9% | 30.7% | 85.5% | 13.2% |
| 7 | Verstappen | 5.9% | 24.5% | 78.6% | 19.6% |

All 22 entries with expected finishing position and P10–P90 bands:
[`results/forecast_2026_14_pre_weekend/predictions.csv`](results/forecast_2026_14_pre_weekend/predictions.csv)

Against a bookmaker board with its 22.7% overround removed, the forecast agrees
at a rank correlation of **0.928** and names the same favourite. Priced at the
quoted odds, quarter-Kelly stakes 0.77% of a bankroll. It is already priced —
agreement with a market is not accuracy.

---

## Quickstart

Python 3.12. The historical panel is committed, so everything below runs offline.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest
```

Forecast a race — an explicit entry list and information cutoff are **required**,
not optional, because a forecast without a stated cutoff cannot be checked:

```bash
.venv/bin/python -m f1pred.cli predict --season 2026 --round 14 \
    --entries configs/entries_2026_14_pre_weekend.csv \
    --cutoff 2026-09-11T01:00:00+00:00
```

<details>
<summary><b>Reproduce the experiments</b></summary>

```bash
# Chronological hierarchical pace comparison, with calibration and bootstrap gates
.venv/bin/python -m scripts.run_state_space_experiment --sims 20000

# Calibrated ranking baselines, Plackett–Luce and corrected practice pace
.venv/bin/python -m scripts.run_experiments

# Strict backtest against snapshots that name the field in advance
.venv/bin/python -m scripts.make_snapshots --seasons 2022 2023 2024 2025
.venv/bin/python -m f1pred.cli backtest --seasons 2022 2023 2024 2025 \
    --snapshots configs/snapshots_2022_2025_pre_weekend.csv --out backtest_tuning_strict
```

`F1PRED_HOME` selects the project data directory when running an installed
package from another working directory.

</details>

---

## Three guards against fooling yourself

Every fitting function takes an explicit `as_of_index` and filters on it, so the
live forecast and the backtest run through **identical code**. On top of that:

| Guard | What it stops |
|---|---|
| **Stage gate** | A feature family is illegal until its snapshot, and an unregistered family fails *closed* |
| **Timestamp gate** | Every forecast must state a cutoff; history is gated on when a result became available, not when it was downloaded |
| **`shifted_expanding`** | The only sanctioned way to build a rolling statistic |

Two real leaks were caught and are now regression-tested: raw results arrive in
finishing order and simulated win probabilities tie at exactly zero deep in the
field, so tie-breaking silently read the answer; and an ingestion gap could
renumber events so a target race's own observations entered its training set.

---

## What has been tried and rejected

Negative results are the point of an ablation, so they are recorded with effect
sizes rather than omitted.

| Candidate | Verdict |
|---|---|
| Corrected FP2 long-run pace (74 weekends ingested) | no validated incremental lift |
| Counted on-track overtakes (20,567 passes) | **structurally inert** — see below |
| Race pace in lap time instead of finishing rank | no effect |
| Posterior strength uncertainty | significantly worse |
| Fitted simulator variance budget | worse |
| Plackett–Luce ranking model | worse on probability, no ranking gain |
| Hierarchical state-space pace model | +2.2pp Top-1, interval spans zero |

Seven serious attempts, and every confidence interval contains zero.

Two findings are worth more than the verdicts. The circuit model's overtaking
index correlates at **−0.26 (p = 0.22)** with actual counted overtakes — it ranks
Imola, one of the hardest circuits in the world to pass at, as the second
*easiest*. And it could not have mattered anyway: at pre-weekend the simulated
grid is drawn from the same latent as race pace, so the blend weight between them
is inert by construction.

The second is in [FINDINGS §7c](docs/FINDINGS.md): a change that **cleared the
adoption gate and was still wrong.** The control arm had been contaminated before
the arms were built. The strict backtest caught it, not the gate.

---

## Evidence and scope labels

Every run writes a manifest with input hashes, the seed, and the command that
produced it. Claims in this README point at files you can open.

- [`results/state_space_experiment/`](results/state_space_experiment/) — the current
  hierarchy comparison: predictions, per-race metrics, variance fits, timing
  coverage, per-outcome calibration, paired intervals, every adoption gate
- [`results/backtest_tuning_strict/`](results/backtest_tuning_strict/) — the 87-race
  strict backtest behind the headline table
- [`results/experiments/`](results/experiments/) — calibrated ranking baselines,
  Plackett–Luce, practice pace

**Read the scope label before reading a number.** Runs marked
`reconstructed_snapshot` name the field in advance but assume availability from
race dates — that is not verified timestamp evidence. Runs marked
`retrospective_conditional_field` use the actual starter list and are refused by
the adoption gate. Neither is proof of live performance.

A change is accepted only if it passes [BUILD_SPEC](docs/BUILD_SPEC.md): full
tests, legal information snapshots, a primary-metric improvement without degraded
calibration, and favourable paired-bootstrap evidence on the tuning window. 2026
has already been inspected, so new confirmation needs unseen races.

---

## What is still open

The incumbent keeps real architectural limitations, and the failed alternatives
above do not resolve them: a rank-transformed race target, hand-set simulator
noise, an observation-count uncertainty formula, confounded team/driver
coefficients, exponential recency weighting in place of a development model.

The state-space candidate in [STATE_SPACE.md](docs/STATE_SPACE.md) addresses
several of these with stronger statistical contracts — teammate contrasts,
random-walk development, empirical-Bayes variance, correlated posterior draws —
and still fails the adoption gate. Full Bayesian hyperparameter uncertainty,
clean-air pace, informative censoring and race-wide incident dynamics remain open.
The available sample does not prove a ceiling; it fails to resolve one.

---

## Layout

| Path | Purpose |
|---|---|
| `f1pred/sources/` | Cached source adapters and ingestion |
| `f1pred/features/` | Strength, circuit, practice and reliability features |
| `f1pred/models/` | State-space hierarchy, retirement hierarchy, ranking calibration |
| `f1pred/sim/` | Monte Carlo simulator |
| `f1pred/evaluation/` | Metrics, chronological comparisons, uncertainty checks |
| `scripts/` | Reproducible ingestion, experiments and reports |
| `tests/` | Offline correctness and statistical tests |
| `data/` | Historical panel and reference inputs |
| `results/` | Executed results and provenance |
| `f1pred/legacy/` | Earlier code; not an accepted forecast feature |

## Documentation

| | |
|---|---|
| [FINDINGS](docs/FINDINGS.md) | Every experiment, effect size and verdict |
| [BUILD_SPEC](docs/BUILD_SPEC.md) | Architecture, data contracts, stage rules, acceptance criteria |
| [STATE_SPACE](docs/STATE_SPACE.md) | The hierarchical candidate: derivation, assumptions, limits |
| [REVIEW](docs/REVIEW_2026-09-11.md) | Independent code review and its ten findings |
| [AUDIT](docs/AUDIT.md) | Audit of the original notebook this replaced |
| [Sources](docs/sources_and_methodology.md) | Data sources and the literature behind the design |
