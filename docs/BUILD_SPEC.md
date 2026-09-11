# F1 Prediction System — Build Specification

Target: calibrated pre-race probabilistic forecasts
Status legend used throughout: **[I]** implemented · **[T]** unit-tested ·
**[B]** backtested · **[V]** statistically validated improvement · **[P]** planned

---

## 1. Purpose and scope

Predict, for each entry in an upcoming Grand Prix, the probability of winning, of
a podium, of a top-10 finish and of retirement, together with an expected
finishing position and an honest uncertainty band — and explain the forecast in
physical and competitive terms.

The system answers *"which driver-and-car combination suits this circuit, under
these conditions, given this reliability and this overtaking environment"*, not
merely *"who has been good"*.

### Non-goals

- Live in-race prediction (the simulator is pre-race only).
- Beating the market. The benchmark is the project's own documented baselines.
- Any forecast that uses information unavailable at its stated prediction stage.

---

## 2. Prediction stages

A forecast is always tagged with the snapshot it was made from. A feature family
may be used only at or after its earliest legal stage.

| Stage | Cut-off | Adds |
|---|---|---|
| `PRE_WEEKEND` | before FP1 | history, championship state, car/driver strength, reliability, track geometry and similarity, issued weather forecast |
| `POST_FP1` | end of FP1 | FP1 corrected pace, first telemetry-derived corner indices |
| `POST_FP2` | end of FP2 | long-run pace, tyre degradation and warm-up, updated forecast |
| `POST_FP3` | end of FP3 | final single-lap pace |
| `POST_QUALI` | end of qualifying | qualifying position and gap, grid after penalties |
| `RACE_START` | lights out | final grid, final forecast |

`POST_RACE` is deliberately **not** a stage: it exists only as a sentinel so that
any feature family not explicitly registered fails closed as illegal.

**Madrid 2026 note.** Round 14 starts 2026-09-13 13:00 UTC; FP1 was 2026-09-11
11:30 UTC. A forecast produced before that is a genuine `PRE_WEEKEND` forecast,
and must not be compared against grid-based numbers, which need qualifying.

---

## 3. Leakage control

Three independent mechanisms, because they catch different mistakes.

1. **Stage gate** — `f1pred.leakage.assert_stage_allowed` checks every feature
   family against `FEATURE_CATALOG`. Unregistered families are illegal.
2. **Timestamp gate** — `assert_observed_before` rejects any row observed at or
   after the prediction cut-off, independent of family.
3. **Shifted aggregation** — `shifted_expanding` is the only sanctioned way to
   build a rolling statistic. It sorts within group, shifts by one, and only then
   aggregates, so race *R* can never enter its own feature.

Structurally, every fitting function takes an explicit `as_of_index` and filters
`event_index < as_of_index`. The live Madrid forecast and the walk-forward
backtest therefore run through **identical code paths** — the backtest is a real
test of the deployed system, not of a parallel implementation.

Forbidden at every pre-race stage: realised race weather, race laps, race pit
stops, race tyre strategy, fastest race lap, incidents, final results, post-race
standings, any later race, end-of-season aggregates.

---

## 4. Data sources

| Source | Used for | Coverage | Access |
|---|---|---|---|
| Jolpica-F1 (Ergast-compatible) | schedule, results, qualifying, standings | 2014–2026, complete | REST, ~0.3 s/req |
| OpenF1 | sessions, laps, stints, pit stops, weather, telemetry | 2023–2026 | REST, ~1.3 s/req, 429-throttled |
| MultiViewer circuit API | centreline geometry, corner markers, pit loss | 32 circuits, older years | REST, requires User-Agent |
| Open-Meteo (forecast + previous-runs) | issued forecasts for archived backtests | global | REST |
| Published circuit specifications | new venues with no telemetry history | as needed | recorded in `data/reference/circuit_specs.csv` with source and retrieval date |

All traffic passes through `f1pred.http_cache.CachedSession`: per-API rate limit,
disk cache keyed by URL hash, and a sidecar provenance record holding the URL,
HTTP status, byte count and UTC fetch time.

### Known source limitations (verified, not assumed)

- **`"Lapped"` is a classified finish.** `positionText` is the authority on
  classification; the status string may only attribute cause.
- **Retirement causes are unavailable from 2024 onward.** Jolpica reports a
  generic `"Retired"`, so a mechanical-versus-incident split is not identifiable
  in the current regime. Total DNF hazard is modelled instead.
- **MultiViewer has no Madrid geometry**, so Madrid's track vector comes from
  published specification plus data-driven analogue transfer.

---

## 5. Architecture

```
                 ┌──────────────────────────────────────────┐
   Jolpica ─────►│ sources/  (typed, cached, provenanced)    │
   OpenF1  ─────►│                                          │
   MultiViewer ─►└────────────────┬─────────────────────────┘
                                  │
                        ┌─────────▼──────────┐
                        │ pipeline.build_panel│  global event_index
                        └─────────┬──────────┘
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
   ┌────────▼────────┐  ┌─────────▼────────┐  ┌─────────▼────────┐
   │ features/pace   │  │features/reliab.  │  │features/circuits │
   │ team + driver   │  │ team + driver    │  │ overtaking diff. │
   │ latent strength │  │ DNF hazard       │  │ attrition, speed │
   └────────┬────────┘  └─────────┬────────┘  └─────────┬────────┘
            └─────────────────────┼─────────────────────┘
                        ┌─────────▼──────────┐
                        │ sim/race           │  Monte Carlo
                        │ quali → grid →     │
                        │ pace → position →  │
                        │ retirement         │
                        └─────────┬──────────┘
                        ┌─────────▼──────────┐
                        │ win / podium / top10 / DNF │
                        │ expected finish, P10–P90   │
                        └────────────────────┘
```

Modular by design: a weak module can be ablated out and measured, and no single
black box is asked to learn everything.

---

## 6. Module specification

### 6.1 `features/pace` — latent strength **[I][T][B]**

Race results are converted to van-der-Waerden normal scores over *classified*
finishers; qualifying is converted from percentage gap to pole. Both land on a
common ±2 scale. A weighted ridge regression with one indicator per constructor
and one per driver decomposes the observations into an additive team effect and a
driver coefficient. Equal ridge penalties do not identify separate causal
car and driver effects; this is an unresolved limitation of the incumbent.

- Retirements are **excluded**, not scored as last: a failure carries no pace
  information, and scoring it last would corrupt a fast-but-fragile car's rating.
- Exponential recency weighting; prior-regime observations additionally
  down-weighted.
- Driver effects shrink toward the team mean by observation weight, so a rookie
  cannot earn an extreme rating from two races.

### 6.2 `features/reliability` — DNF hazard **[I][T][B]**

Total DNF probability, decomposed with hand-set shrinkage into a constructor hazard
(shared by both cars) and a driver excess term in log-odds, with
Beta shrinkage toward the field base rate. A circuit multiplier scales the
hazard; circuits with no history fall back to a layout-family multiplier
(street circuits measured at ~1.19× the calendar average, permanent ~0.93×).

### 6.3 `features/circuits` — venue behaviour **[I][T][B]**

The incumbent uses an inferred track-position persistence proxy: per circuit, the median
grid-to-finish Spearman correlation among classified finishers and the median
absolute position change, each rank-normalised across the calendar and averaged.

The proxy ranked Suzuka 0.98, Jeddah 0.92 and Monaco 0.89, with Imola 0.08.
Its agreement with subsequently measured on-track passes was poor (Spearman
−0.262, p = 0.22; FINDINGS §7b). These values must not be presented as validated
measurements of overtaking difficulty. The counted-pass alternative remains an
experiment without demonstrated predictive lift.

Lap length and average lap speed are **derived from the data** — regulated race
distance ÷ winner's lap count, divided by pole time — rather than hand-copied.
Validation: recovers Monza at 5755 m (actual 5793) and Monaco at 3333 m (actual
3337) once Monaco's 260 km race-distance exception is applied.

Similarity uses **standardised** distance. Raw cosine similarity between
all-positive physical vectors is near 1 for every pair and is not usable.

### 6.4 `sim/race` — Monte Carlo **[I][T][B]**

Each simulated race draws, in order: qualifying performance → grid (only when the
grid is unknown, which is the correct behaviour before qualifying); race pace;
a finishing-order score blending grid advantage and race pace with the circuit's
overtaking weight, plus start-phase and strategy swings priced in grid slots; and
retirements with a correlated race-level chaos shock so attritional races retire
several cars at once.

Grid slots are mapped to the latent scale with the same normal-score transform
used for results, so the blend weight means what it says.

### 6.5 Corner and telemetry layer **[I][T]** — not yet backtested

Preserved in `f1pred/legacy/features.py`: curvature, corner segmentation,
braking efficiency `a = (v_entry² − v_apex²)/(2·d_brake)`, traction index, exit
amplification, lateral g, aero commitment, wind projection onto corner heading.
These are unit-tested but have **not** been shown to improve any forecast, and
are not claimed to.

---

### 6.6 Hierarchical state-space candidate **[I][T][B]** — experimental

See [STATE_SPACE.md](STATE_SPACE.md) for the measurement model, identification
constraints, random-walk dynamics, empirical-Bayes variance fitting, contrast
covariance, retirement hierarchy and actual test protocol. This candidate uses
Q1 and valid lead-lap elapsed times without a rank/fastest-lap fallback. Fitted
variance parameters remain conditional estimates; their uncertainty is not
integrated. Its statistical design tests and predictive adoption gates are
reported separately. An unsuccessful experiment does not resolve the six
architectural criticisms of the incumbent.

## 7. Evaluation protocol

Race-level only. Row-level accuracy is meaningless when each race is one ranking
problem with one winner.

Metrics: winner Top-1 · winner-in-Top-3 · NDCG@5 · Spearman over classified
finishers · rank MAE · winner log-loss · Brier · expected calibration error.

Uncertainty: Wilson intervals on winner accuracy, and **paired** bootstrap over
shared races for every model comparison — pairing removes race difficulty, which
matters enormously at 13–24 races per season.

### Validation design

Strictly chronological walk-forward. For race *R*, fit on everything before *R*.
No random splits across driver rows, ever.

- **Tuning:** 2022–2025 (ground-effect regime, ~92 races).
- **Held-out test:** 2026 (new regulations, concept drift by construction).
- Hyper-parameters are chosen on the tuning window. The originally held-out
  2026 results have since been inspected and cannot serve as untouched evidence
  for changes informed by them. New confirmation requires unseen races.

### Baselines that must be reported

| Baseline | Stage | Why |
|---|---|---|
| championship order | `PRE_WEEKEND` | the honest naive competitor - with the previous season's final order as the opener prior, never a flat zero |
| last race result | `PRE_WEEKEND` | pure recency |
| rolling 5-race mean finish | `PRE_WEEKEND` | reproduces the archived `PRE_WEEKEND_FORM` |
| grid position | `POST_QUALI` | **reference only** — bounds what a pre-weekend model could achieve; comparing a Stage-A forecast to it is a category error |

---

## 8. Acceptance criteria

A change is accepted only if it clears all four:

1. Passes the full pytest suite.
2. Contains no stage or timestamp leakage.
3. Improves at least one primary metric on the **tuning** window without
   degrading calibration.
4. Its paired bootstrap against the incumbent has the mass of the difference on
   the favourable side — a point estimate alone is not evidence at n≈20.

A candidate that fails stays experimental or is redesigned. Rejection is not
proof that the incumbent's architecture is correct. Report paired block
sensitivity for temporally dependent races, and separate statistical correctness
from predictive improvement. The state-space runner applies these gates to
actual artifacts and records every failed condition.

---

## 9. Reproducibility

- One seed (`config.SEED`) threads through every stochastic step.
- Every external byte is cached with a provenance record.
- `pyproject.toml` declares the dependency set; `requirements-lock.txt` pins the
  exact versions that produced the checked-in results.
- Each run writes a manifest with input hashes, settings, seed and command.
- Result artefacts are written only by code that actually ran — no hard-coded
  status tables (see `docs/AUDIT.md`, finding 3).

---

## 10. Roadmap, honestly staged

| Item | Status | Evidence |
|---|---|---|
| Latent strength model | **[B] backtested, no ranking advantage shown** | on a strict 87-race snapshot it ties the championship baseline exactly on Top-1 (43 each) and no metric separates them; the earlier ordering claim was an artifact of a degenerate opener baseline (FINDINGS §1) |
| Monte Carlo simulator | **[B] backtested, partly unproven** | beats every informative probability baseline (championship +0.226, CI [+0.070, +0.406]) but ties a temperature-calibrated softmax over the same strengths (+0.005, CI [-0.030, +0.041]); contributes nothing measurable to ranking |
| Reliability hazard | **[B] backtested, calibration only** | tuning 12.05% predicted against 11.17% observed, 2026 13.85% against 17.56%; it is what makes retirement probabilities exist, and it improves winner log-loss (1.449 against 1.470), but it does not improve ranking |
| Qualifying observations | **[B] measured improvement; snapshot evidence incomplete** | removing them costs 0.030 NDCG, CI [+0.006, +0.056] - the only component with a favourable ranking interval in that comparison |
| Overtaking difficulty | **[B] inert at Stage A, unhelpful at Stage B** | the simulated grid is drawn from the same strength as the pace, so the blend weight cannot change a pre-weekend ordering - the measured-vs-inferred difference is exactly zero. At POST_QUALI a counted pass rate is marginally worse than the inferred proxy (FINDINGS §7b) |
| POST_QUALI stage | **[B] backtested with reconstructed availability** | strict 2026 snapshot: matches the grid baseline Top-1 (0.667, 8/12) and beats it on Spearman (0.838 against 0.798) - the only headline that survives strict scoring intact |
| Practice long-run pace (Stage B) | **[B] measured, not adopted** | 74 of 92 weekends ingested, 49 evaluable; no primary metric clears the gate (FINDINGS §7) |
| Plackett-Luce ranking model | **[B] rejected** | winner log-loss +0.516, CI [+0.401, +0.625], and no ranking gain |
| Track-car compatibility | **[B] rejected** | monotonically worse as the kernel narrows; reproduces the circuit-history failure of the original notebook |
| Dual qualifying/race skills | **[B] rejected** | NDCG -0.017, CI [-0.033, -0.003] |
| Calibration layer | **[B] rejected** | negligible in-sample gain; see the caveat in FINDINGS §8 |
| Time-varying team development | **[I][T] not adopted** | +3.3pp Top-1 inside noise, NDCG fell |
| Corner-level telemetry vectors | **[I][T]** | formulas unit-tested in `f1pred/legacy/`; never backtested |
| Tyre degradation model | **[I][T]** | stints are now ingested with the practice laps; unused |
| Weather and corner wind | **[I][T]** | needs archived issued forecasts |

---

### Availability evidence correction

`race_date + 1 day` is a conservative reconstruction, not a verified publication
timestamp. Forecasts using this proxy are labelled `reconstructed_snapshot`;
verified `published_at` values take precedence and are checked against the cutoff.
Previously saved runs named `strict` use reconstructed historical availability.
The directory name alone does not establish timestamp verification. Actual-starting
fields remain `retrospective_conditional_field` and cannot clear adoption.

### Earlier architecture experiment correction

The saved `results/architecture_experiment/` variance-budget comparison fitted
2024–2025 and applied that fit to 2022–2023. Disjoint seasons do not satisfy
chronological validation. Its adoption verdict is invalidated; the replacement
runner fits only earlier data and writes `results/state_space_experiment/`.

## 11. Measured results summary

See `docs/FINDINGS.md` for full detail, uncertainty and negative results.

| Window | Stage | Races | Top-1 | NDCG@5 | Spearman | Log-loss |
|---|---|---:|---:|---:|---:|---:|
| 2022-2025 tuning, strict | PRE_WEEKEND | 87 | 0.494 | 0.732 | 0.714 | 1.431 |
| 2022-2025 tuning, conditional | PRE_WEEKEND | 92 | 0.500 | 0.734 | 0.707 | 1.451 |
| 2026 holdout, strict | PRE_WEEKEND | 12 | 0.167 | 0.608 | 0.778 | 1.912 |
| 2026 holdout, strict | POST_QUALI | 12 | 0.667 | 0.803 | 0.838 | 1.204 |

On the tuning window the system ties the championship baseline on winner
accuracy, and on the 2026 holdout it loses to it, 2 winners against 5. The
winner was inside the predicted top three in 7 of 13 races and the ordering of
the rest of the field held up, so the failure is concentrated at the top of the
order - the hardest part, and the part the headline metric measures. Everything
recovers once qualifying is known. The system supplies probabilities. Informative calibrated ranking baselines
are required for comparison; see FINDINGS §3 and §5. A low pooled ECE does not
by itself establish calibration across outcomes or conditions.
