# F1 Predictor — Results Summary

## A. Measured 2025 results from your uploaded run

| Feature family | Top-1 | Winner Top-3 | Spearman | Rank MAE | NDCG@5 |
|---|---:|---:|---:|---:|---:|
| Grid baseline | 66.7% | 95.8% | 0.651 | 3.344 | 0.935 |
| Pre-weekend form | 25.0% | 70.8% | 0.545 | 4.092 | 0.869 |
| Form + circuit | 25.0% | 54.2% | 0.538 | 4.146 | 0.864 |
| Qualifying only | 66.7% | 95.8% | 0.640 | 3.453 | 0.932 |
| Qualifying + form | 62.5% | 100.0% | 0.645 | 3.403 | 0.915 |
| Full strict pre-race | 54.2% | 100.0% | 0.659 | 3.320 | 0.922 |
| **Post-practice bridge** | **70.8%** | **100.0%** | **0.663** | **3.298** | 0.921 |

Interpretation: practice information is the strongest lead in the current run, but the one-race Top-1 improvement over grid is too small to call conclusive. The new earlier raw telemetry features are specifically intended to replace crude FP position with corrected single-lap/long-run pace and corner-level car/driver characteristics.

## B. 2026 regime stress test from your uploaded run

The supplied stress file contains **11 races**.

- Grid baseline Top-1: **72.7%**.
- Pre-weekend + circuit: **36.4%**.
- Qualifying + form: **63.6%**.
- Full strict pre-race: **72.7%**.

This reinforces the need to down-weight stale historical form during a regulation regime shift and emphasize current-car telemetry, current-weekend performance and track–car compatibility.

## C. earlier implementation validation

**30/30 tests pass.** See:

- `results/validation_tests_expanded.csv`
- `results/module_status.csv`
- `results/pytest_junit.xml`
- `images/validation_summary.png`

The suite covers corner geometry, braking, traction, aero proxy, wind, car/track vectors, driver/car decomposition, temperature/wet skill, tyres, track evolution, practice pace, dirty air, overtaking, reliability, strategy, weather alignment, leakage guards, data-source helpers and Monte Carlo consistency.

## D. What has NOT yet been claimed

The project does **not** claim that raw earlier corner/weather/tyre features already improve Top-1 beyond 70.8%. The conversation uploads do not contain enough raw history to perform that backtest here. The included Kaggle notebook is designed to perform the next honest experiment: chronological feature-family ablations on the public raw-data sources.
