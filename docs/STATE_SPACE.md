# Hierarchical pace experiment

This experiment addresses the statistical design separately from the question
of predictive improvement. The default simulator remains an unvalidated
incumbent. Its failed replacements do not establish that its architecture is
correct, and a wide interval does not establish a ceiling on achievable accuracy.

## Model and interpretation

For entry *i* on team *t*, the expected log-time score is

```
team_lineup_pace[t] + driver[i] - mean(driver[current_team_lineup])
```

Team pace therefore includes the average contribution of the lineup. Driver
contrasts sum to zero within the current lineup. This specifies an estimand that
the available comparisons can inform. It does not identify a causal pure-car
rating or the absolute skill of drivers who never share connected teammates.
Identical teammates with identical observations receive identical estimates,
even when their team is much faster than another team.

The team and driver states have separate Gaussian population priors and separate
random-walk innovation variances. Each race advances the states; missing races
increase uncertainty. Constructors restart at regulation boundaries, while driver
contrasts persist. That reset is an explicit modelling assumption. New entities
receive a population prior at their first appearance; preallocating their names
must produce the same result as discovering them sequentially.

Six shared variance components are fitted by Kalman innovation marginal
likelihood: initial team and driver variation, team and driver development,
qualifying measurement noise, and race measurement noise. Two fixed optimizer
starts check numerical sensitivity. Positive bounds are numerical safeguards;
convergence and boundary hits are saved for every fold. There is no additional
rookie penalty, observation-count shrinkage, or hand-set start/strategy noise.

This is **hierarchical empirical Bayes**. The full state covariance is retained,
but uncertainty about the fitted variance parameters is not integrated. A
Gaussian prior and likelihood also remain substantive assumptions. Kalman
filtering and likelihood evaluation follow the linear state-space formulation
described in the [statsmodels documentation](https://www.statsmodels.org/stable/statespace.html).
The motivation for explicit centering and symmetric group priors is discussed
in [Stan's sum-to-zero case study](https://mc-stan.org/learn-stan/case-studies/sum_to_zero_vector.html).

## Measurements

Each timing observation is `-100 * log(seconds)`. Differences are in log-percentage
units, approximately percentage differences for small gaps. An orthonormal
Helmert contrast removes the unknown session intercept before each update.
It supplies *n − 1* contrasts, rather than pretending every pair of cars is an
independent observation.

- Qualifying uses Q1 times from a common segment. It does not compare one
  driver's wet Q3 with another driver's dry Q1 or count three segments as three
  independent weekend labels.
- Race observations use valid elapsed race time divided by completed laps for
  classified lead-lap cars. Lapped cars, retirements and missing times are
  excluded. There is no fastest-lap or finishing-rank fallback.
- Elapsed race time includes traffic, stops, neutralisations and shortened races.
  It measures an outcome-related pace proxy, not clean-air or fuel-corrected pace.
  Q1 also contains traffic and changing conditions. Neither is noise-free.
- Selection into lead-lap classification is informative. This model does not
  correct that censoring. Coverage is reported rather than described as 100%.

## Predictive uncertainty

If `S = H P Hᵀ` is the entry covariance, the variance relevant to a comparison is

```
Var(strength[i] - strength[j]) = S[i,i] + S[j,j] - 2*S[i,j]
```

The sampler retains cross-entry covariance and removes the common field offset.
It adds the fitted race measurement variance once, then draws a correlated
Gaussian pace vector. Adding a common uncertainty component to every entry
therefore leaves ranking uncertainty unchanged. Independently drawing marginal
standard errors does not have this property.

A separate beta-binomial hierarchy estimates retirement rates. Historical
constructor-by-regime counts estimate the population prior; current-regime
counts update each team's posterior. Drivers on a team share a sampled latent
rate. This integrates uncertainty about the team rate, which must not be
interpreted as a model of race-wide incidents. Driver-specific retirement effects
and race-wide incident dynamics remain unmodelled.

The simulation conditions on at least one classified finisher, consistent with
scoring a Grand Prix with a winner. Retirement marginals use the same condition.
Retired entries follow finishers, with their relative order approximated by pace.
Only classified finishers count toward podium and top-ten probabilities.

## Chronological experiment

Run from the repository root:

```bash
.venv/bin/python -m pytest --junitxml=results/tests.xml
.venv/bin/python -m scripts.run_state_space_experiment --sims 20000
.venv/bin/python -m scripts.run_state_space_experiment --report-only
```

The `scripts.run_architecture_experiment` entry point runs the same experiment.
Annual variance fits use all earlier races only; states update after a race is
predicted. The 2021 predictions supply calibration warm-up. Evaluation is confined
to 2022–2025. A separate winner-only temperature calibration uses earlier
out-of-fold predictions, never the current race. It does not claim calibrated
podium or retirement probabilities.

The incumbent and official championship baseline are regenerated on the same
races, field, seed and simulation count. All fields are explicitly
`retrospective_conditional_field`. Such comparisons can diagnose a candidate;
they cannot satisfy the publication-time snapshot gate for adoption. The already
inspected 2026 results do not select this model, and future confirmation needs
races that have not informed development.

The runner saves predictions, per-race scores, annual variance fits, per-race
retirement fits, calibration histories, timing coverage and timing-contrast
interval checks. Calibration is reported separately for each model and outcome.
The usual paired race bootstrap is accompanied by four- and eight-race circular
block bootstraps within seasons. These check sensitivity to persistent car form;
they do not remove model-selection bias or create additional seasons of evidence.

Acceptance retains BUILD_SPEC §8: full tests, legal information snapshots,
primary-metric improvement, no calibration deterioration, and favourable paired
bootstrap evidence. The additional block sensitivity must support the claimed
improvement. Any comparison failing a gate remains experimental.

## Evidence and open issues

Actual numerical results and gate decisions are in
[`results/state_space_experiment/`](../results/state_space_experiment/).
`variance_folds.json` records optimizer diagnostics;
`timing_validation.csv` measures out-of-sample contrast coverage;
`acceptance.json` records every gate. The report reads actual pytest XML rather
than manufacturing passing test rows.

The mathematical contracts are tested in
[`tests/test_state_space.py`](../tests/test_state_space.py). They include batch
Gaussian posterior agreement, synthetic interval coverage, common-offset
invariance, teammate identification, temporal changes, regulation resets,
new-entry consistency and future-data exclusion. Synthetic coverage validates
inference under the stated model; it cannot establish real-world coverage or
resolve missing hyperparameter uncertainty.

The incumbent still uses rank targets, hand-set spreads and a confounded ridge
decomposition. This candidate implements a different architecture. Keeping the
incumbent until an alternative passes acceptance does not close those findings.
