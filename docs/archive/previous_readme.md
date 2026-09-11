# F1 Predictor

Calibrated pre-race probabilities (win / podium / top-10 / DNF / expected finish)
for Formula 1, built on real data with strict chronological validation.

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/f1pred ingest                                   # cache 2014-2026 history
.venv/bin/f1pred backtest --seasons 2026                  # walk-forward test
.venv/bin/f1pred predict --season 2026 --round 14          # forecast a race
.venv/bin/pytest -q                                        # 117 tests
```

## Headline result

Pre-weekend (before FP1) forecast, walk-forward, never fitted on the race being
predicted:

| Model | Stage | Races | Top-1 | Top-3 | NDCG@5 | Spearman | Rank MAE | Winner log-loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **This system** | PRE_WEEKEND | 92 | **51.1%** | 79.3% | **0.737** | **0.707** | **3.734** | **1.442** |
| championship order | PRE_WEEKEND | 92 | 45.7% | 77.2% | 0.691 | 0.670 | 3.937 | 2.988 |
| rolling 5-race form | PRE_WEEKEND | 92 | 46.7% | 70.7% | 0.694 | 0.663 | 3.980 | 2.988 |
| last race result | PRE_WEEKEND | 92 | 40.2% | 69.6% | 0.670 | 0.534 | 4.364 | 2.988 |
| *grid position* | *POST_QUALI* | 92 | *55.4%* | *87.0%* | *0.796* | *0.729* | *3.440* | *2.988* |

Tuning window 2022–2025. The grid row is a **reference, not a competitor**: it
uses qualifying, which a pre-weekend forecast does not have. Getting within
4.3 points of it without seeing a single lap of the weekend is the real result.

The log-loss column is the largest gap: baselines rank drivers but produce no
probabilities, so they score the uniform 1/20 (log 20 = 2.996). This system's
1.442 comes from an actually calibrated simulator — **expected calibration error
0.005** on the tuning window and 0.011 on the 2026 holdout.

## Madrid 2026 — the target case

Round 14, the Madring's first Grand Prix, forecast before FP1. A brand-new
circuit has no history, so its behaviour is transferred from the circuits it most
resembles in derived speed-profile space (Miami, Baku, Sochi, Marina Bay).

| Driver | Win | Podium | Top 10 | DNF |
|---|---:|---:|---:|---:|
| Kimi Antonelli | 28.0% | 62.6% | 92.0% | 6.3% |
| George Russell | 19.5% | 50.2% | 82.8% | 13.6% |
| Lando Norris | 14.7% | 43.4% | 81.7% | 14.6% |
| Lewis Hamilton | 13.7% | 44.1% | 90.9% | 6.8% |
| Charles Leclerc | 9.2% | 32.6% | 78.3% | 16.9% |
| Oscar Piastri | 7.9% | 30.8% | 82.7% | 13.2% |
| Max Verstappen | 5.8% | 24.0% | 74.2% | 19.6% |

Against the bookmaker board with the 22.7% overround removed: Spearman **0.928**,
same favourite, mean absolute disagreement **0.89 percentage points**.
Full comparison in `results/madrid_2026_model_vs_market.csv`.

## Why this is not just another gradient-boosted table

1. **It simulates races.** 100,000 Monte Carlo races per forecast, each drawing a
   qualifying result (the grid is *unknown* before Saturday, so it is sampled,
   not assumed), race pace, start and strategy swings, and correlated
   retirements. Probabilities come out of the simulation rather than a softmax.
2. **Pace and reliability are separate models.** A fast, fragile car and a slow,
   reliable one are different objects; folding them together is what made
   retirements the dominant error source in the previous version.
3. **Retirements are excluded from pace estimation**, not scored as last place. A
   blown engine says nothing about how quick the car was.
4. **The circuit is a physical object, not a category.** Overtaking difficulty is
   *measured* from grid-to-finish behaviour; lap length and average speed are
   *derived* from race distance and lap counts (recovering Monza at 5755 m
   against 5793 actual, Monaco at 3333 m against 3337).
5. **Driver effects are team-mate-identified.** A driver is rated against the
   person in the same car, so a strong car cannot manufacture a strong driver.

## Leakage control

Three independent guards, because they catch different mistakes: a **stage gate**
(feature families are illegal until their snapshot, and unregistered families fail
closed), a **timestamp gate**, and `shifted_expanding`, the only sanctioned way to
build a rolling statistic.

Structurally, every fitting function takes an explicit `as_of_index` and filters
on it, so the live forecast and the backtest run through *identical code*.

A leak was found and fixed during development: raw results arrive in finishing
order, and simulated win probabilities tie at exactly zero deep in the field, so
tie-breaking silently read the answer. It is now regression-tested
(`test_tied_scores_do_not_leak_the_finishing_order`).

## What has been shown *not* to work

Reported because negative results are the point of an ablation:

| Idea | Verdict |
|---|---|
| Separate qualifying / race driver skills | **Rejected** — significantly worse NDCG (−0.017, CI [−0.033, −0.003]) |
| Probability calibration layer | **Rejected** — simulator is already calibrated; fitted temperature made the holdout worse |
| Separate team / driver recency half-lives | **Not adopted** — +3.3pp Top-1 is inside noise, NDCG fell |
| Circuit overtaking model at Stage A | **Retained but unproven** — no measurable Stage-A benefit; expected to matter at POST_QUALI |

Qualifying observations, by contrast, help significantly (NDCG +0.025,
CI [0.000, 0.051]) and are kept.

## Documentation

- `docs/BUILD_SPEC.md` — architecture, data contracts, stage rules, acceptance criteria
- `docs/AUDIT.md` — verification of the previous version, including two data-quality defects
- `docs/FINDINGS.md` — every experiment run, with effect sizes and verdicts
- `docs/archive/original_readme.md` — the previous README, preserved

## Layout

```
f1pred/
  config.py          paths, seeds, regimes, tuned hyper-parameters
  http_cache.py      throttled, provenanced, on-disk HTTP cache
  leakage.py         stage gate, timestamp gate, shifted aggregation
  pipeline.py        panel assembly, circuit resolution, forecasting
  forecast.py        upcoming-race path (entry list from the last completed round)
  cli.py             ingest / backtest / predict
  sources/           Jolpica adapter (typed, cached)
  features/          pace, reliability, circuits, compatibility
  models/            calibration
  sim/               Monte Carlo race simulator
  evaluation/        metrics, backtest, ablation, market comparison
f1pred/legacy/                 preserved earlier telemetry/corner layer (unit-tested, unbacktested)
```

`f1pred/legacy/` and its 30 passing tests are kept intact as the previous baseline.
