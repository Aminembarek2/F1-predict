# Verification audit of the original notebook

This audits the project as it was received: a notebook plus a summary script. The
follow-up review of the rewrite that replaced it is in
[REVIEW_2026-09-11.md](REVIEW_2026-09-11.md), with complete backtest reruns and
executable probes.

Audited 2026-09-11 against the repository as received. Every claim below was
checked by running the code, not by reading it.

## What was true

| Claim in `README.md` | Verdict | Evidence |
|---|---|---|
| 30/30 pytest checks pass | **Confirmed** | `pytest -q tests/test_modules_pytest.py` → `30 passed in 6.57s` |
| The corner-geometry formulas are correct on synthetic input | **Confirmed** | curvature of a 100 m circle recovers 1/100 to 5e-4; wind projection, braking and traction identities hold |
| 2025 results are from a prior run, not re-derived | **Confirmed, and honestly labelled** | `run_summary.json` carries an explicit `evidence_boundary` field |
| Grid baseline 66.7%, post-practice 70.8% on 24 races | **Consistent internally** | reproduced exactly from `data/reference/`; the Wilson interval for 16/24 is [0.467, 0.820] |

The project was honest about its own limits. It did not claim the corner-geometry
features had been shown to help.

## What was not true, or not working

1. **The system had never touched real data.** The summary script only re-summarised
   CSVs shipped in `data/reference/`. There is no path from any data source to a
   prediction. `f1pred/legacy/session_pipeline.py` requires `fastf1`, which is not in the
   environment and was never executed.

2. **It could not predict a future race at all.** No module accepts an upcoming
   event. The stated goal — a Madrid forecast — was unreachable from this code.

3. **Fabricated test reporting.** The summary script wrote
   `results/module_status.csv` and `results/validation_tests_expanded.csv` as
   hard-coded `PASS` rows (lines 81–89), unconditionally, whether or not pytest
   ran. Two of the project's own reporting artefacts were therefore not evidence.
   This is the most serious finding and those writers have been removed.

4. **The 2026 stress test is stale.** It covers 11 races; 13 rounds of 2026 are
   complete as of the audit date, and the file carries no provenance or fetch
   timestamp.

## Methodological defects found in the corner-geometry code

- `nearest_tracks` and `cosine_compatibility` apply cosine similarity to raw,
  all-positive feature vectors. Every pair of such vectors has similarity near 1,
  so "track similarity" carried almost no information. Fixed by standardising
  before measuring distance (`f1pred/features/circuits.nearest_circuits`).
- `corner_metrics` derives the braking distance from the first braking sample in
  a fixed ±window, which can pick up the *previous* corner's braking event; and
  it slices `g.loc[apex_i:apex_i, "time_s"]` by label on a possibly non-monotonic
  index.
- `tyre_warmup_laps` scans from the first lap and returns 1 whenever the opening
  lap is already within tolerance, so it does not measure warm-up.
- `wet_skill_residual` subtracts a car mean computed from the same rows that
  include the driver, with no shrinkage and no sample-size guard.
- `monte_carlo_race` samples DNFs independently with no race-level correlation,
  and finishing order ignores the circuit entirely — so P5 at Monaco and P5 at
  Monza produce the same forecast.

## Engineering issues

- No packaging (`pyproject.toml`), no CLI, no logging, no configuration module.
- `f1pred/legacy/__init__.py` re-exports with `import *`, and the test module does the
  same, so the namespace is unauditable and name collisions are silent.
- Statements compressed onto shared lines with semicolons throughout, against the
  project's own stated requirement of typed, documented modules.
- The caching HTTP helper in `data_sources.py` is never called by anything.

## Data-quality defects found while rebuilding

Both were discovered by sanity-checking derived quantities against known reality,
and both would have silently corrupted any reliability model:

1. **`"Lapped"` is a classified finish, not a failure.** Treating the status
   string as authoritative produced a 50.3% DNF rate for 2026. Using
   `positionText` as the authority on classification and the status string only
   for cause attribution gives 17.6%, which is plausible for a new-regulation
   season. 388 rows were affected.
2. **Cause attribution is absent from 2024 onward.** Jolpica reports a generic
   `"Retired"` for essentially every modern DNF (2022: 63 DNFs across 9 distinct
   causes; 2026: 47 of 49 are `"Retired"`). A mechanical-versus-incident split is
   therefore **not identifiable in the current regime**, and the rebuilt
   reliability model estimates total DNF hazard with a team/driver decomposition
   instead of relying on cause labels.

## What was kept

The corner-geometry formulas in `f1pred/legacy/features.py` that are unit-tested and
correct — curvature, braking, traction, wind projection, exit amplification,
lateral g — were retained as the corner-analysis layer. `f1pred/legacy/` and its passing
test suite are left in place as the preserved earlier baseline; nothing was deleted.


## Architecture follow-up

The original audit above is historical. Current architecture work is specified
in [STATE_SPACE.md](STATE_SPACE.md); its executed evidence is in
[results/state_space_experiment](../results/state_space_experiment/).

| Finding | Incumbent status | Candidate treatment |
|---|---|---|
| Seven hand-set simulator parameters | Unresolved | Six pace variance components fitted on earlier races; retirement prior fitted separately; no hand-set start/strategy/chaos terms |
| Rank-transformed race target | Unresolved | Q1 and classified lead-lap elapsed-time contrasts; missing/censored coverage reported |
| Observation-count uncertainty formula | Unresolved | Full covariance propagated to entry contrasts and correlated draws |
| Confounded team and driver effects | Unresolved | Team effect explicitly includes lineup mean; driver effects are centered teammate contrasts; pure causal car/driver decomposition remains unidentified |
| No development dynamics | Unresolved | Separate team and driver random walks; regulation resets and new-entry initialization tested |
| Hand-set point-estimate shrinkage | Unresolved | Hierarchical empirical Bayes; state uncertainty retained, hyperparameter uncertainty remains unmodelled |

Additional correctness findings:

- The old variance-budget experiment applied a 2024–2025 fit to 2022–2023.
  Its verdict is invalid for chronological adoption. The old runner is archived
  and disabled; `scripts.run_architecture_experiment` now runs the chronological
  state-space comparison.
- Historical availability inferred as race date plus one day is now labelled
  `reconstructed_snapshot`. Actual `published_at` evidence overrides the proxy
  and is enforced by the cutoff gate. Download times remain provenance.
- Batch allocation of driver names must not age their priors before debut.
  Filtering with preallocated names and growing the roster sequentially now
  agrees in both posterior means and covariance.
- Candidate podium/top-ten probabilities count classified finishers only.
  Sampling and displayed retirement marginals both condition on a race with at
  least one finisher. The incumbent's rare all-retired/few-finisher edge cases
  remain outside this candidate's correctness claim.

A negative predictive result leaves the incumbent findings open. Passing a
synthetic posterior test establishes a mathematical contract under its assumed
model; it does not establish real-world calibration, causal identification, or
predictive superiority. The full pytest XML and comparison gates are generated
from execution, in accordance with original audit finding 3.
