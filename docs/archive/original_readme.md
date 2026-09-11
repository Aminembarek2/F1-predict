# F1 Predictor  / Corner-Aware

This project upgrades the original F1 predictor from a mostly grid/qualifying model into a modular **Driver × Car × Track × Environment** system.

## What is implemented

The package contains production-style functions for:

1. corner geometry / curvature,
2. corner segmentation,
3. braking efficiency,
4. traction,
5. exit-speed amplification,
6. downstream straight time-gain approximation,
7. aero commitment,
8. corner performance residuals,
9. track demand vectors,
10. car capability vectors with shrinkage,
11. track–car compatibility,
12. nearest-track transfer for unseen circuits,
13. teammate-adjusted driver/car decomposition,
14. corner headwind/crosswind,
15. leakage-safe archived weather hooks,
16. temperature sensitivity,
17. wet-skill residuals,
18. tyre degradation,
19. tyre warm-up,
20. fuel correction,
21. track-evolution correction,
22. single-lap practice pace,
23. long-run practice pace,
24. dirty-air penalty,
25. overtaking probability,
26. reliability/DNF priors and classifier hooks,
27. pit-loss / undercut features,
28. XGBoost LambdaMART grouped race ranking,
29. winner-probability calibration,
30. Monte Carlo win/podium/top-10/DNF simulation.

`results/formula_catalog.csv` gives the formula/method, earliest legal prediction stage, and source for each feature family.

## Validation

- **30/30 pytest checks pass** on the implementation suite.
- Tests cover telemetry formulas, missing-value paths, leakage controls, source normalization, telemetry/weather alignment, reliability/strategy logic, and Monte Carlo probability consistency.
- `results/pytest_junit.xml` contains the machine-readable test report.

## Empirical results available from the uploaded run

The conversation supplied a real **24-race 2025 holdout** and an **11-race 2026 regime stress-test summary**. Those results are preserved under `data/reference/` and summarized in `results/`.

For 2025:

- grid baseline Top-1 winner accuracy: **66.7% (16/24)**,
- best existing bridge experiment (`POST_PRACTICE_FULL`): **70.8% (17/24)**,
- bridge winner-in-Top-3: **100% (24/24)**,
- bridge Spearman: **0.663**,
- bridge rank MAE: **3.298**,
- bridge NDCG@5: **0.921**.

Important: that 70.8% result comes from the already-run practice/qualifying/form experiment, **not yet from the new raw corner/weather/tyre telemetry features**. The raw earlier feature lift must be measured by running the Kaggle notebook against the public telemetry/weather APIs. No raw-telemetry result is fabricated in this package.

## Evidence boundary

The uploaded CSVs do not contain the full multi-season raw telemetry, archived forecasts, interval history, tyre stints and FIA component corpus. Therefore the package deliberately separates:

- **measured predictive results** from your existing 2025/2026 run, and
- **implementation validation** of the new earlier telemetry modules.

The included Kaggle notebook is the reproducible path to measure each new feature family chronologically.

## Leakage policy

Prediction snapshots are explicit:

- `PRE_WEEKEND`
- `POST_FP1`
- `POST_FP2`
- `POST_FP3`
- `POST_QUALI`
- `RACE_START`

Current-race race laps, realized race weather, current-race pit stops/stints and final outcomes are blocked before race start. Historical weather backtests should use archived forecasts that were actually available before the prediction timestamp.

## Main files

- `archive/original/f1_predictor.ipynb` — self-contained Kaggle notebook.
- `f1pred/legacy/features.py` — core telemetry/features.
- `f1pred/legacy/session_pipeline.py` — FastF1 session → corner/practice feature pipeline.
- `f1pred/legacy/aggregation.py` — driver/circuit/capability aggregation.
- `f1pred/legacy/modeling.py` — LambdaMART, DNF classifier, calibration, Monte Carlo.
- `f1pred/legacy/leakage.py` — feature-stage and timestamp leakage guards.
- `f1pred/legacy/data_sources.py` — OpenF1 / Open-Meteo / TracingInsights helpers and caching.
- `tests/test_modules_pytest.py` — 30-test suite.
- `scripts/summarize_legacy_results.py` — regenerates local result summaries/charts from the uploaded CSVs.
- `research/sources_and_methodology.md` — methodology and source notes.

## Run locally

```bash
pip install -r requirements.txt
pytest -q tests/test_modules_pytest.py
python scripts/summarize_legacy_results.py
```

## Run on Kaggle

1. Upload `archive/original/f1_predictor.ipynb` or the complete ZIP.
2. Enable internet for the first public-data cache build.
3. Run the embedded validation cells.
4. Select years/events and prediction snapshot.
5. Cache FastF1/OpenF1 data once.
6. Run chronological ablations.
7. Train pace/ranking and DNF models separately.
8. Calibrate on a prior season.
9. Simulate 10,000+ races.
10. Download the generated `f1_predictor_results.zip`.
