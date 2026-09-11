# F1 Predictor

Calibrated pre-race probabilities — win, podium, top ten, retirement, expected
finishing position — for Formula 1, built from real data with strict
chronological validation.

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev,telemetry]"
.venv/bin/python -m f1pred.cli ingest                   # cache 2014-2026 history
.venv/bin/python -m scripts.make_snapshots --seasons 2026        # name the field first
.venv/bin/python -m f1pred.cli backtest --seasons 2026 \
    --snapshots configs/snapshots_2026_2026_pre_weekend.csv
.venv/bin/python -m scripts.make_entry_snapshot --season 2026 --round 14
.venv/bin/python -m f1pred.cli predict --season 2026 --round 14 \
    --entries configs/entries_2026_14_pre_weekend.csv \
    --cutoff 2026-09-11T01:00:00+00:00
.venv/bin/pytest -q                                     # 155 tests
```

## How it works

![Architecture: sources to panel to leakage gates to three fitted models to the simulator to published probabilities](images/architecture.svg)

Regenerate with `python -m scripts.make_architecture` — the counts in it are read
from the panel, so the picture cannot describe a system that no longer exists.

## What it does, and what that is worth

Before a wheel has turned on a race weekend, it forecasts the race: 100,000
Monte Carlo races, each drawing a qualifying result (the grid is genuinely
unknown before Saturday, so it is sampled rather than assumed), race pace, start
and strategy swings, and correlated retirements. Probabilities come out of the
simulation, not a softmax.

On the 2022–2025 tuning window, forecast before FP1, naming the field in advance
from the previous round and scored on the field it named — 87 races, season
openers excluded because there is no previous round to name them from:

| Model | Top-1 (95% Wilson) | NDCG@5 | Spearman | Rank MAE | Winner log-loss |
|---|---|---:|---:|---:|---:|
| this system | 49.4% [39.2, 59.7] | 0.732 | 0.714 | 3.703 | **1.431** |
| championship standings | 49.4% [39.2, 59.7] | 0.733 | 0.702 | 3.769 | — |
| rolling 5-race form | 47.1% [37.0, 57.5] | 0.701 | 0.677 | 3.943 | — |
| last race result | 39.1% [29.5, 49.6] | 0.675 | 0.543 | 4.361 | — |

**On winner accuracy it ties the championship table** — 43 out of 87 each — and
no metric separates them: Top-1 0.000 [−0.069, +0.069], Spearman +0.012 [−0.009,
+0.034]. An earlier draft of this README claimed a significant ordering
advantage. That was an artifact of a baseline that returned a flat zero at season
openers, where nobody has scored yet; given the previous season's final order
instead, the gap closes. [`docs/FINDINGS.md`](docs/FINDINGS.md) §1 shows the
arithmetic.

On the held-out 2026 season it did worse still: 2 winners in 12 races against the
championship table's 5, while keeping its ordering edge and its calibration.
Once qualifying is known it recovers to 66.7% and ties the grid reference — the one headline that survives strict scoring intact.

**What is actually worth something here is the probabilities**, which no ranking
baseline produces at all.

## The probabilities

Ranking baselines emit no probabilities, so comparing against them on log-loss
is a comparison the simulator cannot lose. Each baseline is therefore also given
a probability by a temperature fitted only on earlier races:

| Probability model | Winner log-loss | Paired vs simulator |
|---|---:|---|
| **simulator** | **1.450** | — |
| calibrated latent strength | 1.455 | +0.005 [−0.030, +0.041] |
| calibrated championship | 1.676 | +0.226 [+0.070, +0.406] |
| Plackett–Luce | 1.966 | +0.516 [+0.401, +0.625] |
| uniform 1/20 | 2.988 | +1.536 [+1.347, +1.720] |

The simulator beats every informative baseline — except a one-parameter softmax
over the same latent strengths, which ties it. The Monte Carlo earns its place
by producing podium, top-ten, retirement and grid distributions that a softmax
cannot produce at all, not by sharpening the win probability.

Calibration is measured per outcome, because a low winner ECE proves nothing on
its own — the uniform forecast scores exactly zero on it. Winner ECE 0.008,
podium 0.019, top ten 0.054, retirement 0.035; 79.8% of finishes land inside the
published P10–P90 band against a nominal 80%. Top-ten probabilities are the
weakest of the four and are the obvious next job.

## Is it worth acting on?

Calibration is a property of a forecast; edge is a property of a forecast *and a
price*. Priced against a real bookmaker board at the quoted odds — never the
de-vigged ones, because the house pays the quoted price:

| | |
|---|---:|
| House margin on turnover | 18.5% |
| Mean disagreement, model vs market | **0.89 pp** |
| Selections with positive expected value | 3 of 22 |
| Quarter-Kelly stake this justifies | **0.77% of bankroll** |

**The forecast is essentially already priced.** It disagrees with the market by
roughly a twentieth of the margin it would have to overcome. That is one board
and one race, so it is not a measurement of long-run edge — but the machinery is
built and tested (`scripts/run_staking.py`), and a corpus can accumulate one
board at a time. [`docs/FINDINGS.md`](docs/FINDINGS.md) §7a.

## Madrid 2026

Round 14, the Madring's first Grand Prix, forecast before FP1 from an entry list
dated six days earlier. A brand-new circuit has no history, so its behaviour is
transferred from the venues it most resembles in derived speed-profile space.

| Driver | Win | Podium | Top 10 | Retirement |
|---|---:|---:|---:|---:|
| Kimi Antonelli | 27.7% | 62.3% | 93.4% | 6.3% |
| George Russell | 19.7% | 51.1% | 86.1% | 13.6% |
| Lando Norris | 14.9% | 44.0% | 85.0% | 14.6% |
| Lewis Hamilton | 13.3% | 43.2% | 92.4% | 6.8% |
| Charles Leclerc | 9.3% | 33.1% | 82.1% | 16.9% |
| Oscar Piastri | 7.9% | 30.7% | 85.6% | 13.2% |
| Max Verstappen | 5.9% | 24.5% | 78.6% | 19.6% |

Against the bookmaker board with the 22.7% overround removed: Spearman 0.928,
same favourite, mean absolute disagreement 0.89 pp. That is agreement, not
accuracy — it means the forecast is not claiming to know something the market
does not.

## A component that was not measuring what it claimed

`overtaking_difficulty` was *inferred* from how much each finishing order
resembled its grid. Counting the passes that actually happened — 20,567 of them
across 82 races — the two agree at Spearman −0.262 (p = 0.22). Imola, one of the
hardest circuits in the world to overtake at, was ranked the second *easiest*.

That reframes the ablation result above: the circuit model contributes nothing to
ranking because the number fed into it is close to noise, not because circuits do
not matter. Details in [`docs/FINDINGS.md`](docs/FINDINGS.md) §7b.

## The whole feature set

Two numbers per driver per race, and nothing else:

| Observation | Source | Count |
|---|---|---:|
| qualifying pace | lap-time gap to the session best, as a percentage | 5,295 |
| race pace | finishing position, rank-transformed to a latent scale | 4,564 |

From those, ridge regression fits a team effect and a team-mate-identified driver
effect; a separate beta-binomial model fits retirement hazard; a circuit profile
sets the weight between track position and pace. Seven numbers per driver reach
the simulator. **No tyres, no weather, no telemetry, no practice.** Every one of
those was either tested and rejected or never shown to pay — see
[`docs/FINDINGS.md`](docs/FINDINGS.md).

The one open improvement that needs no new data: race pace is rank-transformed,
so the model learns that a car finished first, never that it won by twenty
seconds. Lap-time margin is already ingested at 100% coverage and unused.

## How it is built

1. **Pace and reliability are separate models.** A fast, fragile car and a slow,
   reliable one are different objects; folding them together is what made
   retirements the dominant error source in the original notebook.
2. **Retirements are excluded from pace estimation**, not scored as last place.
   A blown engine says nothing about how quick the car was.
3. **The circuit is a measured object, not a category.** Overtaking difficulty
   is estimated from grid-to-finish behaviour; lap length and average speed are
   derived from race distance and lap counts, recovering Monza at 5755 m against
   5793 actual and Monaco at 3333 m against 3337.
4. **Driver effects are team-mate-identified** — a driver is rated against the
   person in the same car. This still does not make them causal: with two
   identical drivers per team, ridge fitting splits them ±0.209 anyway.

## Leakage control

Three independent guards, because they catch different mistakes: a **stage gate**
(a feature family is illegal until its snapshot, and an unregistered family fails
closed), a **timestamp gate** on an explicit cutoff that every forecast must
supply, and `shifted_expanding`, the only sanctioned way to build a rolling
statistic. Past results are gated on when they were *published*, not when this
repository downloaded them — re-fetching a 2019 race today does not make its
outcome future information.

Every fitting function takes an explicit `as_of_index` and filters on it, so the
live forecast and the backtest run through identical code.

Two real leaks were found and are now regression-tested: raw results arrive in
finishing order and simulated win probabilities tie at exactly zero deep in the
field, so tie-breaking silently read the answer; and an ingestion gap could
renumber events so that a target race's own observations entered its training
set.

A strict backtest names the field in advance from the previous round and is
scored on the field it named: a driver named who then did not start is scored
behind everyone who did, a starter nobody named is excluded and counted, and a
race whose winner was never named is dropped as unscoreable. Runs that instead
read the field off the finishing results are labelled
`retrospective_conditional_field` and refused by the adoption gate. The
difference between the two turns out to be at most 0.002 on any metric.

## Documentation

- [`docs/FINDINGS.md`](docs/FINDINGS.md) — every experiment, with effect sizes and verdicts
- [`docs/BUILD_SPEC.md`](docs/BUILD_SPEC.md) — architecture, data contracts, stage rules
- [`docs/REVIEW_2026-09-11.md`](docs/REVIEW_2026-09-11.md) — independent review; its ten findings and their status
- [`docs/AUDIT.md`](docs/AUDIT.md) — audit of the original notebook this replaced
- [`docs/sources_and_methodology.md`](docs/sources_and_methodology.md) — data sources and the literature behind the design

## Layout

```
f1pred/
  config.py          paths, seeds, regimes, tuned hyper-parameters
  http_cache.py      throttled, provenanced, on-disk HTTP cache
  leakage.py         stage gate, timestamp gate, shifted aggregation
  pipeline.py        panel assembly, circuit resolution, forecasting
  forecast.py        upcoming-race path
  cli.py             ingest / backtest / predict
  sources/           Jolpica results; OpenF1 and timing-archive laps, overtakes, pit stops
  features/          pace, reliability, circuits, compatibility, practice
  models/            calibration, rank-to-probability models
  sim/               Monte Carlo race simulator
  evaluation/        metrics, backtest, ablation, candidate experiments, market
  legacy/            archived corner-geometry layer: unit-tested, never backtested,
                     and contributing to none of the results above
scripts/             ingestion, ablation, experiments, snapshots, figures
```
