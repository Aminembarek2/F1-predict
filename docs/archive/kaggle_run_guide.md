# Kaggle run guide

Use `archive/original/f1_predictor.ipynb`.

Recommended first full backtest:

- Training history: 2023–2024 for raw telemetry-rich features.
- Calibration: 2024 or rolling expanding-window calibration.
- Main holdout: 2025.
- Regime stress test: available 2026 completed races, reported separately.
- Prediction snapshots: PRE_WEEKEND, POST_FP3, POST_QUALI.

Ablation order:

1. grid baseline,
2. qualifying gap,
3. corrected practice pace,
4. corner braking/traction/aero,
5. track–car compatibility,
6. driver corner skill,
7. archived forecast + corner wind,
8. tyre degradation/warm-up priors,
9. dirty-air/overtake priors,
10. DNF/reliability,
11. strategy uncertainty,
12. Monte Carlo outcomes.

Never tune on the 2025/2026 holdout after inspecting its labels. Use expanding historical folds for feature selection and hyperparameters.
