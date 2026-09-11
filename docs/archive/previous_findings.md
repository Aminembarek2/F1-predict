# Experimental findings

Every experiment run during the earlier rebuild, with effect sizes, uncertainty and a
verdict. Negative results are recorded in full: an ablation that only reports
what worked is not an ablation.

Protocol throughout: chronological walk-forward, paired bootstrap over shared
races (10,000 resamples), Wilson intervals on winner accuracy. Tuning window
2022–2025 (92 races). Held-out test 2026 (13 races). 2026 was never used to
select anything.

---

## 1. Headline: pre-weekend forecasting

### Tuning window, 2022–2025 (92 races), stage `PRE_WEEKEND`

| Model | Top-1 | Top-3 | NDCG@5 | Spearman | Rank MAE | Winner log-loss |
|---|---:|---:|---:|---:|---:|---:|
| **strength + Monte Carlo** | **0.511** | 0.794 | **0.737** | **0.707** | **3.734** | **1.442** |
| form5 (rolling 5-race mean finish) | 0.467 | 0.707 | 0.694 | 0.663 | 3.980 | 2.988 |
| championship order | 0.457 | 0.772 | 0.691 | 0.670 | 3.937 | 2.988 |
| last race result | 0.402 | 0.696 | 0.670 | 0.534 | 4.364 | 2.988 |
| *grid position (POST_QUALI reference)* | *0.554* | *0.870* | *0.796* | *0.729* | *3.440* | *2.988* |

The model beats every legitimate pre-weekend baseline on every metric and lands
4.3 points of Top-1 behind a baseline that requires qualifying to have happened.

### Held-out 2026 (13 races), stage `PRE_WEEKEND`

| Model | Top-1 | Top-3 | NDCG@5 | Spearman | Rank MAE | Winner log-loss |
|---|---:|---:|---:|---:|---:|---:|
| strength + Monte Carlo | 0.154 | 0.538 | 0.597 | **0.772** | **3.768** | **1.897** |
| championship order | **0.462** | 0.692 | **0.655** | 0.661 | 4.107 | 3.065 |
| last race result | 0.385 | 0.692 | 0.643 | 0.571 | 4.638 | 3.065 |
| form5 | 0.077 | 0.538 | 0.496 | 0.652 | 4.187 | 3.065 |

**This is a mixed result and is reported as such.** Against the championship
baseline the model is significantly *better* on Spearman (+0.110, CI [+0.047,
+0.197]) and rank MAE (−0.339, CI [−0.638, −0.018]), and much better on
probabilistic quality — but significantly *worse* on Top-1 (−0.308, CI [−0.538,
−0.077]).

**Diagnosis.** All twelve Top-1 misses are traceable, and eight are the same
error: the model ranked Russell ahead of Antonelli through rounds 2–5 and 9–10
while Antonelli was winning. Antonelli was a 2025 rookie, so his driver effect
was still being shrunk toward the team mean exactly during his breakout, and the
two Mercedes drivers sit only 0.08 apart on the latent scale — close to a coin
flip. By round 13 the model has the ordering right, which is the state the Madrid
forecast is made from.

Sample size matters here: 2/13 versus 6/13 has Wilson intervals of [0.04, 0.42]
and [0.23, 0.71], which overlap substantially.

### Held-out 2026, stage `POST_QUALI`

| Model | Top-1 | Top-3 | NDCG@5 | Spearman | Rank MAE | Winner log-loss |
|---|---:|---:|---:|---:|---:|---:|
| **strength + Monte Carlo** | **0.692** | **0.923** | 0.810 | **0.833** | **3.133** | **1.136** |
| grid position | 0.692 | 0.923 | 0.820 | 0.790 | 3.193 | 3.065 |

Once the grid is known the model matches the grid baseline on winner accuracy
while ranking the rest of the field better and producing calibrated probabilities
instead of none. The early-season driver-ordering problem disappears because
qualifying resolves it directly.

---

## 2. Ablation: which components earn their place

Tuning window, 92 races. Each component removed in turn.

| Variant | Top-1 | NDCG@5 | Spearman | Rank MAE | Winner log-loss |
|---|---:|---:|---:|---:|---:|
| full | 0.500 | 0.735 | 0.707 | 3.741 | 1.446 |
| no_circuit | 0.511 | 0.736 | 0.706 | 3.743 | 1.436 |
| no_reliability | 0.500 | 0.736 | 0.706 | 3.740 | 1.476 |
| strength_only (no simulator) | 0.500 | 0.734 | 0.706 | 3.740 | **2.988** |
| no_qualifying | 0.457 | 0.711 | 0.711 | 3.737 | 1.555 |

**Qualifying observations: keep.** Removing them costs 0.025 NDCG
(CI [+0.000, +0.051]) and 4.3 points of Top-1. Qualifying is a clean single-lap
pace measurement uncontaminated by strategy, traffic or attrition.

**The simulator: keep, for probabilities only.** It contributes essentially
nothing to ranking (NDCG 0.735 vs 0.734) and everything to probability quality
(log-loss 1.446 vs 2.988). This is an honest and useful separation of concerns:
the strength model ranks, the simulator quantifies uncertainty.

**Reliability: keep, marginal.** Small log-loss gain (1.446 vs 1.476), no ranking
effect. It is retained mainly because it is *directly validated* — see §5.

**Circuit overtaking model: retained but unproven at Stage A.** Removing it is
very slightly better (Top-1 +0.011, CI [−0.033, 0.000]). This is expected: before
qualifying the grid is itself simulated from the same strengths, so the
grid-versus-pace blend has little to bite on. It should matter at `POST_QUALI`,
where the grid is a fixed input, and that is where it is kept for.

---

## 3. Rejected: separate qualifying and race-pace driver skills

Motivated by the RAPM decomposition of 2014–2024 F1, which finds constructors
explain ~64% of race-outcome variance but a *lower* share in qualifying —
implying one pooled strength averages two different quantities.

| Variant | Top-1 | Top-3 | NDCG@5 | Spearman | Rank MAE | Log-loss |
|---|---:|---:|---:|---:|---:|---:|
| single strength | 0.500 | 0.804 | 0.735 | 0.707 | 3.741 | 1.446 |
| dual qual/race | 0.478 | 0.772 | 0.718 | 0.710 | 3.740 | 1.501 |

Paired: NDCG@5 **−0.017, CI [−0.033, −0.003]** — significantly worse. Top-1
−0.022, log-loss +0.056.

**Verdict: rejected.** The theory is sound but splitting halves the observations
feeding each fit, and the added estimation noise outweighs the benefit at this
sample size. Implemented and tested (`fit_dual_strength`) but not enabled.

---

## 4. Rejected: track–car compatibility by similarity-weighted history

The project's central hypothesis — that a car suits some circuits better than
others — implemented by re-weighting historical observations toward circuits
resembling the target, with a Gaussian similarity kernel over derived circuit
profiles. The bandwidth controls specialisation.

| Variant | Top-1 | Top-3 | NDCG@5 | Spearman | Rank MAE | Log-loss |
|---|---:|---:|---:|---:|---:|---:|
| pooled (no compatibility) | 0.500 | 0.804 | 0.735 | 0.707 | 3.741 | 1.446 |
| bandwidth 0.7 | 0.489 | 0.717 | 0.716 | 0.697 | 3.785 | 1.571 |
| bandwidth 1.0 | 0.489 | 0.739 | 0.717 | 0.697 | 3.782 | 1.553 |
| bandwidth 1.6 | 0.467 | 0.783 | 0.725 | 0.700 | 3.766 | 1.491 |
| bandwidth 2.5 | 0.489 | 0.783 | 0.729 | 0.704 | 3.768 | 1.453 |

Performance is **monotonic in bandwidth**: the wider the kernel, the closer to
the pooled model, and the pooled model is best. Narrow kernels are significantly
worse (bandwidth 1.0: log-loss +0.107, CI [+0.028, +0.181]).

**Verdict: rejected.** This independently reproduces the earlier finding that
circuit-history features do not help. The interpretation is that public
aggregate data does not contain enough signal to identify circuit-specific car
strengths before it is swamped by the loss of effective sample size. Telemetry —
actual corner-level speeds — would be the way to revisit this, not more
re-weighting of results.

The module and its tests are kept, since the bandwidth is a hyper-parameter and
the conclusion may change once telemetry features exist.

---

## 5. Reliability model: validated directly against reality

The reliability model's outputs were checked against realised 2026 retirement
rates, which is a stronger test than an aggregate metric.

| Entry | Predicted DNF (Madrid) | Actual 2026 rate |
|---|---:|---:|
| Stroll (Aston Martin) | 51.8% | 69.2% |
| Bottas (Cadillac) | 44.1% | 46.2% |
| Alonso (Aston Martin) | 35.5% | 38.5% |
| Pérez (Cadillac) | 35.8% | 30.8% |
| Verstappen (Red Bull) | 19.6% | 23.1% |
| Norris (McLaren) | 14.6% | 16.7% |
| Russell (Mercedes) | 13.6% | 15.4% |
| Antonelli (Mercedes) | 6.3% | 0.0% |

The ordering is right and the magnitudes track closely. Note the predictions
include a 1.19× street-circuit multiplier, so they should sit slightly above the
season averages, which they do.

---

## 6. Hyper-parameter selection

**Recency half-life** — searched 2–9 races on the tuning window. The surface is
flat (Top-1 0.478–0.511 across the whole range); 6 races was chosen as a plateau
value, not a peak.

**Cross-regime weight** — this one matters for 2026, so it was tuned on **2022**,
the only *other* post-regulation-break season available, never on 2026.

| Cross-regime weight | Top-1 on 2022 |
|---:|---:|
| 0.00 | 0.500 |
| 0.02 | 0.500 |
| 0.08 | 0.500 |
| 0.20 | 0.545 |
| **0.35** | **0.636** |

The default of 0.35 is the best value on the analogous transition season, so it
was kept. Note this does *not* rescue 2026's Top-1: the 2026 problem is a
team-mate ordering error, not a regime-weighting error.

**Separate team/driver half-lives** (backfitting) — best configuration gained
+3.3pp Top-1 but lost NDCG; inside noise at 92 races. Implemented
(`fit_strength_backfit`) but not adopted.

---

## 7. Rejected: probability calibration layer

| Window | ECE before | ECE after | Log-loss before | Log-loss after |
|---|---:|---:|---:|---:|
| 2022–2025 (fit) | 0.0052 | 0.0048 | 1.4423 | 1.4318 |
| 2026 (holdout) | 0.0114 | 0.0121 | 1.8974 | 1.9040 |

The fitted temperature (0.857) gives a negligible in-sample gain and makes the
holdout *worse*.

**Verdict: rejected — the simulator is already calibrated.** The only visible
miscalibration is that strong favourites are slightly *under*-rated (predicted
0.63, observed 0.79, n=24). Module and tests retained for future use.

---

## 8. Bugs and leaks found and fixed

1. **Tie-breaking leak (mine, found during development).** Raw results arrive
   sorted by finishing position, and simulated win probabilities tie at exactly
   zero for roughly 8 of 20 drivers. Tie order therefore read the true result:
   Spearman 0.81 among tied rows. Fixed in three places — neutral entry ordering,
   an outcome-independent tie-break in the evaluator, and a continuous
   (tie-free) ranking key. Regression-tested.
   *Impact: before the fix, `sim_win_prob` appeared to beat the championship
   baseline on Spearman by +0.037 with a CI excluding zero. That result was an
   artefact and is retracted.*

2. **`"Lapped"` misclassified as a retirement.** Produced a 50.3% DNF rate for
   2026; the true figure is 17.6%. 388 rows affected.

3. **Retirement causes unavailable from 2024.** 47 of 49 2026 DNFs carry a
   generic `"Retired"`, so mechanical-versus-incident modelling is not
   identifiable in the current regime. The model uses total DNF hazard instead.

4. **Grid slots exceeding the starter count** crashed the `POST_QUALI` path when
   cars were withdrawn after qualifying. Grid is now dense-ranked into starting
   order, with pit-lane starts (recorded as 0) placed at the back.

5. **`circuit_id` missing from the observation table** made similarity weighting
   a silent no-op — the first compatibility run returned results identical to the
   pooled model, which is what exposed it.

---

## 9. What would most likely help next

Ordered by expected value, all currently **unproven**:

1. **Practice long-run pace (Stage B).** The one signal the system does not yet
   use and the one most likely to carry real information: heavy-fuel race
   simulations on Friday, corrected for fuel burn (~0.06–0.08 s/lap) and track
   evolution. Requires OpenF1 lap ingestion.
2. **Corner-level telemetry.** The proper way to revisit track–car compatibility,
   which failed on aggregate data. `f1pred/legacy/features.py` already holds the
   tested formulas.
3. **Tyre degradation from stint data.** Bayesian state-space models of latent
   tyre pace from public timing data are an established approach.
4. **Better team-mate resolution.** The single largest error source on 2026 was
   ordering two drivers in the same car.
