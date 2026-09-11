# Findings

**Architecture update:** the current chronological hierarchy and its limitations
are documented in [STATE_SPACE.md](STATE_SPACE.md). Results are generated under
[`results/state_space_experiment/`](../results/state_space_experiment/).
The old §7c variance-budget comparison reversed training/evaluation chronology;
it cannot authorize adoption. “Strict” historical runs below use reconstructed
availability, not verified publication-time snapshots.


Every experiment run against real data, with its effect size and its verdict.
Negative results are recorded in the same detail as positive ones, because the
point of an ablation is to find out what does *not* pay for itself.

Written 2026-09-11, against the code in this commit. Reproduce any row with the
command printed beside it; the run manifests under `results/` carry the input
hashes, the seed and the dependency versions that produced these numbers.

**Read the scope label before reading a number.** A run marked
`retrospective_conditional_field` takes its entry list from the finishing results
of the race being predicted, so the model is told who started and who was
withdrawn. A run marked `timestamped_snapshot` names the field in advance from
the previous round and is scored on that field, whether or not it was right.
§1 reports both. Ablations and candidate experiments stay conditional, because
they compare variants with each other rather than with reality, and the adoption
gate refuses them on that basis.

---

## 1. Where the system stands

Tuning window 2022–2025, forecast before FP1, never fitted on the race being
predicted. Two runs: one that is handed the starter list, and one that has to
name the field in advance from the previous round.

```bash
python -m scripts.make_snapshots --seasons 2022 2023 2024 2025
python -m f1pred.cli backtest --seasons 2022 2023 2024 2025 --sims 20000 \
    --snapshots configs/snapshots_2022_2025_pre_weekend.csv --out backtest_tuning_strict
```

**Strict, 87 races** — season openers have no previous round to copy and are
dropped; one further race is unscoreable because its winner was never named.

| Model | Top-1 (95% Wilson) | NDCG@5 | Spearman | Rank MAE | Winner log-loss |
|---|---|---:|---:|---:|---:|
| this system | 49.4% [39.2, 59.7] | 0.732 | 0.714 | 3.703 | **1.431** |
| championship standings | 49.4% [39.2, 59.7] | 0.733 | 0.702 | 3.769 | — |
| rolling 5-race form | 47.1% [37.0, 57.5] | 0.701 | 0.677 | 3.943 | — |
| last race result | 39.1% [29.5, 49.6] | 0.675 | 0.543 | 4.361 | — |

**It ties the championship table.** Exactly: 43 winners each out of 87. Paired,
the Top-1 difference is 0.000 [−0.069, +0.069], NDCG −0.002 [−0.032, +0.029],
Spearman +0.012 [−0.009, +0.034], Rank MAE −0.066 [−0.185, +0.050]. Nothing is
significant on any metric.

### Where the advantage went

An earlier version of this document claimed a significant ordering advantage.
It was an artifact, and it is worth recording exactly how it worked, because
neither cause had anything to do with the model:

| Run | Model Top-1 | Championship Top-1 | Gap |
|---|---:|---:|---:|
| 92 races, conditional field, flat-zero opener baseline | 50.0% | 46.7% | +3.3 pp |
| 92 races, conditional field, **fair** opener baseline | 50.0% | 48.9% | +1.1 pp |
| 87 races, conditional field (openers dropped) | 49.4% | 49.4% | 0.0 pp |
| 87 races, **strict** snapshot | 49.4% | 49.4% | 0.0 pp |

1. **The baseline was broken at season openers.** At round one nobody has
   scored, so the championship baseline returned a flat zero and the evaluator
   tie-broke it alphabetically. Four races per tuning window were therefore free
   wins. It now falls back to the previous season's final standings, which is
   information any person reading a championship table in March would have.
2. **The rest was the openers themselves**, which the strict run cannot forecast
   at all and drops.

The strict field costs almost nothing — compare rows three and four, which differ
by at most 0.002 on any metric. So review finding 4 was real but not inflationary:
the conditional runs were not cheating their way to a better number. That is
worth knowing, and it is not what I expected to find.

What survives is narrow and honest: **the system's probabilities are worth
something (§3), its ordering is level with reading the championship table, and
its winner accuracy is not better than reading the championship table.**

## 2. The 2026 holdout, where it went wrong

```bash
python -m scripts.make_snapshots --seasons 2026
python -m f1pred.cli backtest --seasons 2026 --sims 20000 \
    --snapshots configs/snapshots_2026_2026_pre_weekend.csv --out backtest_2026_strict
```

| Run | Races | Model Top-1 | Championship Top-1 | Model NDCG@5 | Model Spearman |
|---|---:|---:|---:|---:|---:|
| PRE_WEEKEND, strict | 12 | 16.7% | 41.7% | 0.608 | 0.778 |
| PRE_WEEKEND, conditional | 13 | 15.4% | 38.5% | 0.597 | 0.771 |
| POST_QUALI, **strict** | 12 | **66.7%** | 41.7% | 0.803 | 0.838 |
| POST_QUALI, conditional | 13 | 69.2% | 38.5% | 0.810 | 0.833 |

**The pre-weekend forecast lost to the championship table on winner accuracy in
the holdout season, 2 winners against 5.** It kept its ordering advantage
(Spearman 0.778 against 0.748) and stayed calibrated, but the headline number the
project was built around collapsed on the first season it had not seen. Twelve
races is a small sample and 2026 is a new regulation regime — both are reasons,
neither is an excuse. The strict field changes nothing here either.

Once qualifying is known the picture reverses, and **this is the one headline
that survives the strict treatment intact**: 66.7% winner accuracy on the strict
run, 8 out of 12, matching the grid reference exactly and beating it on Spearman
(0.838 against 0.798) while losing marginally on NDCG@5 (0.803 against 0.810).
Knowing the grid is worth roughly 50 points of winner accuracy to this model —
far more than anything in the pre-weekend feature set.

## 3. Are the probabilities worth anything?

The previous version compared its probabilities only against a uniform 1/20
forecast, which every ranking baseline scores identically because none of them
emits a probability at all. That comparison cannot fail. Informative baselines
were built instead: each ranking baseline's scores are turned into probabilities
by a temperature fitted **only on earlier races**, chronologically, with a 2021
warm-up.

```bash
python -m scripts.run_experiments --sims 10000
```

| Probability model | Winner log-loss | Paired vs simulator | Verdict |
|---|---:|---|---|
| **simulator** | **1.450** | — | incumbent |
| calibrated latent strength | 1.455 | +0.005 [−0.030, +0.041] | **ties the simulator** |
| calibrated championship | 1.676 | +0.226 [+0.070, +0.406] | beaten |
| calibrated 5-race form | 1.755 | +0.304 [+0.150, +0.463] | beaten |
| Plackett–Luce | 1.966 | +0.516 [+0.401, +0.625] | beaten |
| calibrated last race | 2.043 | +0.592 [+0.352, +0.863] | beaten |
| uniform 1/20 | 2.988 | +1.536 [+1.347, +1.720] | beaten |

Two conclusions, and the second is the uncomfortable one:

1. The simulator's probabilities really are better than informative baselines,
   not merely better than a constant. That claim now has a paired test behind it.
2. **A one-parameter softmax over the same latent strengths matches it.** The
   100,000-race Monte Carlo buys nothing measurable in winner log-loss over a
   temperature-calibrated ranking of the strengths it is fed. The simulator's
   remaining justification is that it also emits podium, top-10, retirement and
   grid distributions, which a softmax cannot produce at all — not that it makes
   the win probability sharper.

Plackett–Luce, suggested as a cheaper alternative in the review, is clearly
worse on probability quality and no better on ranking. **Rejected.**

## 4. Component ablation

Each component switched off in turn, same races, same seed.

```bash
python -m scripts.run_ablation --sims 8000
```

These variants run at 8,000 simulations per race and §3 at 10,000, against the
20,000 of §1, which is why the full pipeline reads 51.1% here and 50.0% there.
One race out of 92 is 1.1 points: the Top-1 metric is thin enough that the
simulation budget moves it, which is its own argument for reading the intervals.

| Removed | Top-1 | NDCG@5 | NDCG@5 cost of removing it | Verdict |
|---|---:|---:|---|---|
| nothing (full) | 51.1% | 0.738 | — | — |
| circuit model | 51.1% | 0.737 | +0.001 [0.000, +0.004] | **unproven** |
| reliability model | 50.0% | 0.735 | +0.003 [−0.001, +0.009] | **unproven for ranking** |
| the simulator itself | 50.0% | 0.734 | +0.004 [0.000, +0.010] | **unproven for ranking** |
| qualifying observations | 45.7% | 0.708 | +0.030 [+0.006, +0.056] | **earns its place** |

Only qualifying observations pay for themselves on ranking. Ranking the field
directly by latent strength, with no simulation at all, is statistically
indistinguishable from the full pipeline — consistent with §3. The reliability
model does improve winner log-loss (1.449 against 1.470 without it), which the
ranking metrics cannot see, and it is what makes the retirement probabilities
exist; it is kept on those grounds and labelled unproven for ranking.

## 5. Calibration, per outcome

A low winner ECE on its own proves nothing: the uniform forecast attains an ECE
of exactly zero under the same bins. Every published probability is therefore
scored separately, with counts, on the strict runs
(`results/backtest_tuning_strict/probability_report.csv`).

| Outcome | Predicted | Observed | Tuning ECE | n | 2026 ECE |
|---|---:|---:|---:|---:|---:|
| winner | 5.04% | 5.04% | **0.008** | 1726 | 0.018 |
| podium | 15.12% | 15.12% | 0.019 | 1726 | 0.051 |
| top 10 | 50.41% | 50.41% | 0.054 | 1726 | 0.094 |
| retirement | 12.04% | 12.40% | 0.035 | 1726 | 0.061 |
| finish inside published P10–P90 | 80% | 79.8% | — | 1726 | 79.0% |

Win probabilities are well calibrated and the P10–P90 band is almost exactly
right. **Top-10 probabilities are the weakest**, at an ECE roughly seven times
the winner's, and they get worse on 2026 rather than better. Retirement
probabilities run 0.4 pp low on tuning and 5.5 pp low on 2026, where actual
attrition reached 19.5%. These are pooled, binned estimates: they depend on the
binning and say nothing about conditional calibration.

This is the part of the system that is genuinely worth something — it is the only
thing here that no ranking baseline produces at all — and it is also the part
with an obvious next job: the top-ten probabilities need work.

## 6. The Madrid forecast

Round 14, the Madring's first Grand Prix, forecast before FP1 from an entry
snapshot dated 2026-09-07 with a cutoff of 2026-09-11T01:00Z. This is the only
result in this document produced under `timestamped_snapshot`: nothing in it
depends on knowing who finished.

```bash
python -m scripts.make_entry_snapshot --season 2026 --round 14
python -m f1pred.cli predict --season 2026 --round 14 \
  --entries configs/entries_2026_14_pre_weekend.csv \
  --cutoff 2026-09-11T01:00:00+00:00 --sims 100000
```

| Driver | Win | Podium | Top 10 | Retirement |
|---|---:|---:|---:|---:|
| Kimi Antonelli | 27.7% | 62.3% | 93.4% | 6.3% |
| George Russell | 19.7% | 51.1% | 86.1% | 13.6% |
| Lando Norris | 14.9% | 44.0% | 85.0% | 14.6% |
| Lewis Hamilton | 13.3% | 43.2% | 92.4% | 6.8% |
| Charles Leclerc | 9.3% | 33.1% | 82.1% | 16.9% |
| Oscar Piastri | 7.9% | 30.7% | 85.6% | 13.2% |
| Max Verstappen | 5.9% | 24.5% | 78.6% | 19.6% |

The circuit has no history, so its behaviour is transferred from the venues it
most resembles in derived speed-profile space. Against the bookmaker board with
the 22.7% overround removed: Spearman 0.928, same favourite, mean absolute
disagreement 0.89 pp (`results/forecast_2026_14_pre_weekend/model_vs_market.csv`).
Agreement with a market is not accuracy — it means the forecast is not saying
anything the market has not already priced.

## 7. Practice pace

FP2 long-run pace, corrected for tyre age, compound and session evolution by
robust regression with driver/stint intercepts, blended into the strength score
by a fixed declared rule with no search over outcomes.

```bash
python -m scripts.ingest_practice --seasons 2022 2023 2024 2025
python -m scripts.run_experiments --sims 10000
```

Coverage: 74 of 92 tuning weekends have FP2 laps. The 18 without are sprint
weekends from 2023 onward, which replaced FP2 with a second qualifying; they are
reported as missing, never imputed. After reserving the first weekends for
chronological calibration, 49 weekends were evaluable.

| Metric | Practice blend | Control (same races) | Difference | Verdict |
|---|---:|---:|---|---|
| Top-1 | 44.9% | 46.9% | −2.0 pp [−8.2, +4.1] | — |
| NDCG@5 | 0.725 | 0.716 | +0.009 [−0.017, +0.036] | — |
| Winner log-loss | 1.489 | 1.482 | +0.007 [−0.077, +0.094] | — |

**Not adopted.** Nothing clears the gate on 49 weekends. Spearman (0.726 against
0.692) and rank MAE (3.54 against 3.72) both moved the right way, which is worth
another experiment at POST_FP2 with stint and traffic controls — but those are
not the pre-declared primary metrics, and reading them as a result would be
exactly the metric-shopping this register exists to prevent.

## 7a. Is any of it worth acting on?

Calibration is a property of a forecast. Edge is a property of a forecast *and a
price*. A forecast can be perfectly calibrated and still worthless to act on,
because the market already knew. This is the test of the second thing.

```bash
python -m scripts.run_staking \
  --forecast results/forecast_2026_14_pre_weekend/predictions.csv \
  --odds data/reference/madrid_2026_market_odds.csv
```

| Quantity | Value |
|---|---:|
| Bookmaker overround | 22.7% |
| House margin on turnover (1 − 1/1.227) | **18.5%** |
| Mean absolute disagreement, model vs market | **0.89 pp** |
| Selections with positive expected value, at the quoted price | **3 of 22** |
| Quarter-Kelly stake the model's own numbers justify | **0.77% of bankroll** |

**The forecast is essentially already priced.** The model disagrees with the
board by 0.89 percentage points on average against a house margin of 18.5 —
roughly twenty times the disagreement. Its own Kelly sizing says to risk under
one percent of a bankroll on this race.

Of the three positive selections, one is Lawson at 151.0 on a 0.34 pp
disagreement. A third of a percentage point producing a +33% expected value is a
warning, not an opportunity: longshot probabilities are the least reliable part
of any forecast, and the Kelly fraction of 0.0022 says so. Strip it out and the
substantive positions are Russell (+8.6%) and Hamilton (+6.8%), together £7.18 of
a £1,000 bankroll.

### The rule this module exists to enforce

Stakes are priced at the **quoted** odds, never at the de-vigged ones. Removing
the overround is correct for comparing probabilities and wrong for comparing
money, because the bookmaker pays the quoted price. On this same board:

| Priced at | Positive-EV selections | Mean EV |
|---|---:|---:|
| quoted odds (correct) | 3 of 22 | −0.607 |
| de-vigged "fair" odds (the trap) | 5 of 22 | −0.518 |

De-vigging first hands back the house margin on paper and manufactures edge
nobody can collect. `test_expected_value_is_priced_at_the_quoted_odds_not_the_fair_ones`
pins it.

### What this does not establish

One board, one race, captured once. It measures *this* forecast against *this*
market, not long-run edge. A real answer needs closing prices for a corpus of
races, and no free historical source exists — the Odds API requires a key and
prices its historical endpoint on a paid tier. The machinery is built and tested,
so the corpus can be accumulated forward, one board captured per race before its
cutoff. Until then, the honest reading is the one above: the disagreement is an
order of magnitude smaller than the margin it would have to overcome.

## 7b. The circuit model measures the wrong thing

The ablation in §4 says removing the circuit model costs nothing: NDCG +0.001
[0.000, +0.004]. The natural reading is that circuits do not matter much for
ranking. That reading is wrong, and a second data source proves it.

`overtaking_difficulty` is **inferred** — from how much each race's finishing
order resembles its grid. A timing feed reports every on-track pass as an event,
so the same quantity can simply be counted.

```bash
python -m scripts.ingest_race_events --seasons 2023 2024 2025 2026
```

**20,567 passes across 82 races and 24 circuits.** The measured ordering is what
anyone who watches the sport would expect: Monaco hardest at 10.2 passes per 100
racing laps, then Imola, Jeddah, Marina Bay, Baku; Bahrain easiest at 40.7.

The inferred index the forecaster actually uses agrees with it at **Spearman
−0.262, p = 0.22** across those 24 circuits. Correct in sign, indistinguishable
from zero in strength. Where it disagrees, it disagrees badly:

| Circuit | Inferred rank | Measured rank | Gap |
|---|---:|---:|---:|
| Imola | 23rd hardest (i.e. easiest) | **2nd hardest** | 21 |
| Shanghai | 6th hardest | 22nd | 16 |
| Catalunya | 4th hardest | 18th | 14 |
| Austin | 20th | 7th | 13 |
| Suzuka | **1st hardest** | 13th | 12 |

Imola is one of the hardest circuits on the calendar to pass at, and the proxy
ranks it the second easiest. That is not noise around a working measure; it is a
measure that does not work.

**This reframes §4.** The circuit model contributes nothing to ranking not
because circuits are unimportant, but because the number being fed into it is
close to noise. The proxy confounds three different things it cannot separate: a
circuit where passing is hard, a race where nobody needed to pass, and a race
where the quick cars started at the front anyway.

Evidence: `results/overtaking_validation.csv`, `results/overtaking_measured.csv`.

### And fixing it does not help

The measured rate is now wired in under the same chronological rule as everything
else — a circuit's value at race *R* uses only races before *R* — and the swap was
run on 2024–2025, where enough earlier measured races exist.

```bash
python -m scripts.run_circuit_experiment --seasons 2024 2025 --stage POST_QUALI
```

| Stage | Metric | Measured − inferred | 95% interval |
|---|---|---:|---|
| PRE_WEEKEND | Top-1, NDCG@5, Spearman | **0.000** | exactly zero |
| POST_QUALI | Top-1 | −2.1 pp | [−8.3, +4.2] |
| POST_QUALI | NDCG@5 | −0.004 | [−0.019, +0.010] |
| POST_QUALI | winner log-loss | +0.041 | [−0.018, +0.102] |

**Not adopted.** A correct measurement makes the forecast very slightly worse,
though nothing is significant on 48 races.

Two things explain this, and the first is the more interesting:

1. **At PRE_WEEKEND the parameter is inert by construction.** The simulated grid
   is drawn from the same latent strength as the race pace, so grid advantage and
   pace are the same quantity plus independent noise. Blending them in any
   proportion returns the same expected ordering — which is why the difference is
   *exactly* zero on three metrics, not merely small. No measurement of circuit
   overtaking, however accurate, can change a pre-weekend forecast in this
   architecture. That is a property of the simulator, not of the data.
2. **At POST_QUALI the lever is live and still does not pay.** `overtaking_blend`
   maps difficulty onto a grid weight spanning 0.19 to 0.87 across the real
   calendar, so the mechanism has ample range. The forecast simply does not
   improve when the weight is set correctly rather than arbitrarily.

A caveat on the measurement itself: a race's 400-odd passes include DRS trains,
lapping and midfield churn. That is the right quantity for "how much overtaking
happens here" and possibly the wrong one for "can the second-quickest car get
past the quickest", which is what decides a winner.

**So the component stays as it is, and is now honestly labelled: an inert
parameter at Stage A, and an unhelpful one at Stage B.** The value of this
exercise was not a better forecast. It was discovering that a number the project
presented as measured was not measured, and that the architecture could not have
used it even if it had been.

`pit_lane_loss` is the same story in miniature: the simulator prices strategy
luck at a flat 2.0 grid slots everywhere, while the measured pit-lane cost ranges
from 18.1 s at Albert Park to well over 25 s elsewhere, from 2,833 real stops.

## 7c. Archived architecture comparison — chronology invalid for adoption

Four changes were implemented to address the standing criticisms of the model:
a physically-meaningful race target, measured estimation uncertainty, a fitted
variance budget, and all three together.

```bash
python -m scripts.calibrate_simulator --seasons 2024 2025
# Archived command, no longer an adoption experiment.
python -m scripts.run_state_space_experiment --sims 20000
```

**All four were rejected.** Every factor, isolated against the true incumbent on
44 earlier races. The variance-budget arms violate chronology because their fit
used 2024–2025. These numbers describe an archived diagnostic, not valid
chronological evidence:

| Arm | Winner log-loss | Difference | 95% interval | Verdict |
|---|---:|---:|---|---|
| incumbent | **1.157** | — | — | retained |
| race pace in lap time | 1.174 | +0.017 | [−0.039, +0.073] | no effect |
| fitted variance budget | 1.376 | +0.219 | [−0.219, +0.737] | worse, not significant |
| posterior strength spread | 1.594 | +0.438 | [+0.264, +0.590] | **significantly worse** |
| all three | 1.522 | +0.365 | [+0.189, +0.527] | **significantly worse** |

Ranking was untouched by every arm — Top-1 identical to the fourth decimal.

### The first run said the opposite, and it was wrong

The first version of this experiment reported the fitted budget as a clear
winner: −0.076 log-loss, interval [−0.094, −0.055], gate cleared. **That result
was an artifact of the experiment, not of the model.** The posterior-spread
change had been wired into the code path before the arms were built, so it was
present in the control as well as the treatments. The comparison measured "fitted
noise given a broken spread" against "chosen noise given a broken spread", and
never tested the spread itself. Isolating every factor against the real incumbent
reversed the sign.

It was caught only because the adopted change was pushed through the full strict
backtest, where log-loss went from 1.431 to 1.640 — a number that could not be
reconciled with an experiment claiming an improvement. **An experiment that
agrees with itself is not evidence; the check that matters is the one run in
different conditions.**

### Why measured uncertainty made it worse

This is the instructive part. The ridge posterior standard error of a driver's
strength is around 0.75 in latent units, against a field spread of 1.43 — genuinely
large, and correctly computed, including the strong negative covariance between
the confounded team and driver effects.

But the simulator does not need the uncertainty of each driver's strength. It
needs the uncertainty of the **contrasts** between them, because a race is a
ranking. Most of that posterior error is common-mode: the whole field's level is
poorly pinned down, and that shifts everyone together, cancelling in any ordering.
Adding the full standard error independently to each entry double-counts it and
inflates every car's spread until the field is mush.

The fitted budget then compounded the error. It was fitted *against* the inflated
spread, so it drove all four noise terms toward zero to compensate — a correct
solution to the wrong problem, which is exactly why it collapsed when the spread
was put back.

The relevant quantity is the full covariance of entry contrasts. The current
state-space experiment implements and tests it; see STATE_SPACE.md. That does
not retroactively repair this archived marginal-spread comparison.

### What stands

The original criticism survives intact: **the simulator's seven variance
parameters are hand-set and were never fitted to anything.** That is still true,
still the reason a one-parameter softmax ties it (§3), and still the most
valuable open problem in the model. It is simply not solved by replacing a chosen
number with a fitted one that is worse. `FITTED_VARIANCE_BUDGET` stays in
`config.py` as a recorded negative result, and
`test_variance_budget_keeps_the_constants_the_evidence_supports` stops the
defaults drifting back to it.

`f1pred/features/racepace.py` is kept for the same reason: measuring race pace in
seconds rather than finishing order is obviously the better-posed quantity, it is
available through a mixture of race-time and fastest-lap fallbacks, and this
implementation changed nothing. The 100% claim was coverage of that mixture,
not coverage of comparable elapsed-race-time observations. That result is worth having
recorded rather than rediscovered.

## 8. Rejected earlier, still rejected

Carried forward from the previous evaluation; not re-run.

| Idea | Verdict |
|---|---|
| Separate qualifying / race driver skills | **rejected** — NDCG −0.017 [−0.033, −0.003] |
| Fitted probability calibration layer | **rejected** — the simulator is already calibrated; a fitted temperature made the holdout worse |
| Separate team / driver recency half-lives | **not adopted** — +3.3 pp Top-1 is inside noise and NDCG fell |
| Track–car compatibility kernel | **rejected** — monotonically worse as the kernel narrows |

The calibration-layer rejection is the weakest of the four: it was argued partly
from deterioration on the 2026 holdout, which the stated protocol says was never
used to select anything. The verdict stands because the in-sample gain was
negligible anyway (log-loss 1.4423 to 1.4318), but the reasoning was not clean,
and §3 now re-tests the same question against informative baselines instead.

## 9. Status of the review findings

The review in `REVIEW_2026-09-11.md` raised ten. Where a fix landed, the test
that pins it is named.

| # | Finding | Status |
|---|---|---|
| 1 | Missing events could leak the target into training | **fixed** — one authoritative event index, validated on every observation table (`test_ingestion_gap_cannot_admit_target_observations`) |
| 2 | Live POST_QUALI ignored the known grid | **fixed** — a later stage without its snapshot is refused, and the grid is passed (`test_postquali_missing_grid_is_an_error`, `test_live_and_backtest_use_the_same_grid_and_seed`) |
| 3 | Displayed DNF probability ≠ simulated marginal | **fixed** — correlated retirements now preserve the advertised marginals; the two agree to Monte Carlo noise, 0.003 at 100k sims (`test_simulated_dnf_rate_matches_the_input_hazard`) |
| 4 | Field membership used information unavailable at the stage | **fixed** — `scripts/make_snapshots.py` names the field in advance and a declared policy scores the forecast on the field it named, including when that was wrong (`test_a_named_driver_who_does_not_start_is_scored_behind_the_field`). §1 reports the strict run; it costs at most 0.002 on any metric |
| 5 | Timestamp enforcement was not wired to production | **partly fixed** — an explicit cutoff is required; inferred historical availability is now labelled reconstructed, actual published_at values take precedence, and entry snapshots must carry their own timestamps (`test_timestamp_gate_runs_in_forecast`, `test_download_time_is_provenance_and_not_a_leak`) |
| 6 | The probability benchmark could not fail | **fixed** — §3 |
| 7 | Validation labels exceeded the evidence | **partly fixed** — current scope corrections are recorded above; §5 reports per-outcome calibration and counts |
| 8 | Championship baseline was cumulative GP points | **fixed** — official standings after the preceding round, sprint points included, and the season opener now falls back to the previous season's final order instead of a degenerate flat zero (`test_season_opener_baseline_uses_the_previous_final_standings`). Repairing this removed two thirds of the model's apparent advantage: see §1 |
| 9 | Displayed rank ≠ evaluated rank | **fixed** — `win_probability_rank` and `predicted_rank` are separate columns, and Top-1 is measured on the ranking actually shown |
| 10 | Reproducibility and provenance gaps | **partly fixed** — one seed throughout, run manifests with input hashes, and checked-in runners for the ablation, the experiments and the entry snapshot, and `requirements-lock.txt` pins the exact environment these numbers came from. |

## 10. What is still not established

- No result here is evidence of live accuracy except §6, which has no outcome yet.
- §7c is the clearest warning in this document: a change was measured, cleared
  the adoption gate, was adopted, and was wrong. The gate only protects against
  the comparisons it is actually given.
- §7a is a single board. It says this forecast is priced in; it does not measure
  long-run edge, and nothing here should be staked on.
- The tuning window chose the hyper-parameters, so its intervals are optimistic.
- 2026 has now been inspected. It is development data from here on; any change
  informed by §2 must be validated on races nobody has looked at.
- Driver effects are not established as causal. In a synthetic probe with two
  identical drivers per team and no transfers, the fitted driver effects still
  split ±0.209 purely from team performance, because drivers and constructors are
  confounded under equal ridge penalties.
- The corner-geometry layer in `f1pred/legacy/` is unit-tested on synthetic input
  and has never been backtested. It contributes nothing to any number above.
