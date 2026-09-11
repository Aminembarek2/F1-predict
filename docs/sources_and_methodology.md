# Sources and methodology — checked 2026-09-11

## Public data layer

- **FastF1** — lap timing, telemetry, weather, session metadata and circuit information. https://docs.fastf1.dev/
- **OpenF1** — historical/recent car data, laps, intervals, stints, pit stops, race control and weather. https://openf1.org/docs/
- **Open-Meteo Previous Runs / Single Runs** — archived model forecasts. This matters because a historical ML backtest should use a forecast that was available *before* the race, not the realized race weather. https://open-meteo.com/en/docs/previous-runs-api
- **FIA documents** — event-specific technical and sporting documents; useful for component-state/penalty features when structured extraction is added. https://www.fia.com/documents
- **TracingInsights 2026** — optional public telemetry mirror/fallback for reproducible session fixtures. https://github.com/TracingInsights/2026

## Modeling choices

### Ranking, not independent rows

A Grand Prix is a grouped ranking problem: the 20-ish drivers compete inside one race. The package keeps XGBoost LambdaMART (`rank:ndcg`) as the main tabular ranking baseline, with one race as one query/group. Current XGBoost documentation explicitly supports this `qid` formulation. https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html

### Separate pace, reliability and strategy

The package does not ask one model to explain every retirement or strategy swing. It builds:

1. expected pace/ranking,
2. reliability / DNF probability,
3. tyre and strategy uncertainty,
4. Monte-Carlo outcome probabilities.

That separation is more interpretable and better aligned with motorsport causality.

### Telemetry sequence models are optional, not the default

Recent work has used deep sequence architectures for pit-stop decision support on FastF1 telemetry (Sasikumar et al., 2025, Frontiers in AI, DOI 10.3389/frai.2025.1673148). That supports using LSTM/TCN-style models for *lap-by-lap strategy*, but it does not imply a neural network is automatically superior for the smaller pre-race tabular ranking problem.

### Strict leakage control

Prediction stages are explicit: PRE_WEEKEND, POST_FP1, POST_FP2, POST_FP3, POST_QUALI, RACE_START. Current-race race laps, realized race weather, pit stops, tyre stints and outcomes are blocked before race start. Archived weather must come from a forecast issuance available before the prediction timestamp.

## Corner-geometry features

The implementation includes:

- geometric curvature from XY telemetry,
- corner entry/apex/exit speed,
- braking distance and implied deceleration,
- traction acceleration,
- exit-speed amplification by downstream straight length,
- lateral-g × throttle aero-commitment proxy,
- low/medium/high-speed circuit demand,
- car capability vectors with shrinkage,
- track–car cosine compatibility,
- nearest-track transfer for new circuits,
- teammate-adjusted driver/car decomposition,
- corner-level headwind/crosswind,
- track-temperature sensitivity,
- teammate/car-adjusted wet skill,
- tyre degradation, warm-up and fuel correction,
- track-evolution corrected practice pace,
- dirty-air penalty,
- overtaking probability,
- reliability priors,
- pit-loss / undercut features,
- Monte Carlo win/podium/top-10/DNF probabilities.

## Evidence boundary

The uploaded conversation artifacts provide a full 2025 holdout output and an 11-race 2026 stress-test summary, but not the raw multi-season telemetry/weather/stint corpus. Accordingly, this package reports the uploaded empirical results separately from the 30 implementation tests. Any lift from the corner-geometry features must be measured by rerunning the archived notebook against the public APIs/caches.
