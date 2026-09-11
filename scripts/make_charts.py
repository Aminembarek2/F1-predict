"""Generate result figures from the artefacts written by the backtests."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from f1pred.config import DATA_DIR, IMAGES_DIR, RESULTS_DIR, ensure_dirs

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
ACCENT, MUTED, WARN = "#1f6feb", "#8b949e", "#d1242f"
FORECAST_RUN = "forecast_2026_14_pre_weekend"
REFERENCE_DIR = DATA_DIR / "reference"


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", IMAGES_DIR / name)


def chart_leaderboard(run: str, name: str, title: str) -> None:
    lb = pd.read_csv(RESULTS_DIR / run / "leaderboard.csv").sort_values("top1")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    y = np.arange(len(lb))
    lo = lb["top1"] - lb["top1_ci_low"]
    hi = lb["top1_ci_high"] - lb["top1"]
    colours = [
        ACCENT if m == "strength_mc" else (WARN if "grid_reference" in m else MUTED)
        for m in lb["model"]
    ]
    ax.barh(y, lb["top1"], color=colours, height=0.6)
    ax.errorbar(
        lb["top1"], y, xerr=np.vstack([lo, hi]), fmt="none", ecolor="#24292f", capsize=3, lw=1
    )
    ax.set_yticks(y, lb["model"])
    ax.set_xlabel("winner Top-1 accuracy (Wilson 95% CI)")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    _save(fig, name)


def chart_calibration() -> None:
    from f1pred.models.calibration import reliability_bins

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for path, label, colour in [
        ("backtest_tuning", "2022-2025 tuning", ACCENT),
        ("backtest_2026_preweekend", "2026 holdout", WARN),
    ]:
        p = pd.read_parquet(RESULTS_DIR / path / "predictions.parquet")
        p = p[p["model"] == "strength_mc"]
        t = reliability_bins(p, prob_col="win_probability", outcome_col="winner")
        ax.plot(t["mean_predicted"], t["observed_frequency"], "o-", color=colour, label=label, ms=5)
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1, label="perfect calibration")
    ax.set_xlabel("predicted win probability")
    ax.set_ylabel("observed win frequency")
    ax.set_title("Win-probability calibration")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "calibration_curve.png")


def chart_ablation() -> None:
    a = pd.read_csv(RESULTS_DIR / "ablation" / "report.csv").sort_values("ndcg5")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, col, label in [
        (axes[0], "ndcg5", "NDCG@5 (ranking quality)"),
        (axes[1], "winner_logloss", "winner log-loss (lower is better)"),
    ]:
        d = a.sort_values(col, ascending=(col == "winner_logloss"))
        colours = [ACCENT if m == "full" else MUTED for m in d["model"]]
        ax.barh(np.arange(len(d)), d[col], color=colours, height=0.6)
        ax.set_yticks(np.arange(len(d)), d["model"])
        ax.set_xlabel(label)
    fig.suptitle("Feature-family ablation, 2022-2025 tuning window", y=1.02)
    _save(fig, "feature_ablation.png")


def chart_circuit_difficulty() -> None:
    from f1pred.cli import _load_panel
    from f1pred.features.circuits import build_circuit_profiles

    # Built from the panel rather than read from a stale CSV, and written out so
    # the figure and the table beside it can never disagree.
    results = _load_panel().results
    prof = build_circuit_profiles(results.loc[results["season"] >= 2018])
    prof.to_csv(RESULTS_DIR / "circuit_profiles.csv", index=False)
    prof = prof.dropna(subset=["overtaking_difficulty"]).sort_values("overtaking_difficulty")
    fig, ax = plt.subplots(figsize=(6.4, 6))
    colours = [WARN if s else ACCENT for s in prof["is_street"]]
    ax.barh(np.arange(len(prof)), prof["overtaking_difficulty"], color=colours, height=0.7)
    ax.set_yticks(np.arange(len(prof)), prof["circuit_id"], fontsize=7)
    ax.set_xlabel("overtaking difficulty (0 = pace decides, 1 = grid decides)")
    ax.set_title("Measured grid-dependence by circuit, 2018-2026")
    handles = [plt.Rectangle((0, 0), 1, 1, color=WARN), plt.Rectangle((0, 0), 1, 1, color=ACCENT)]
    ax.legend(
        handles, ["street / temporary", "permanent"], frameon=False, fontsize=8, loc="lower right"
    )
    _save(fig, "circuit_overtaking_difficulty.png")


def chart_madrid(run: str = FORECAST_RUN) -> None:
    fc = pd.read_csv(RESULTS_DIR / run / "predictions.csv").head(10).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    y = np.arange(len(fc))
    axes[0].barh(y - 0.2, fc["win_probability"] * 100, height=0.38, color=ACCENT, label="win")
    axes[0].barh(
        y + 0.2, fc["podium_probability"] * 100, height=0.38, color="#a5c8ff", label="podium"
    )
    axes[0].set_yticks(y, fc["driver_code"])
    axes[0].set_xlabel("probability (%)")
    axes[0].set_title("Madrid 2026: win and podium")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].barh(y, fc["dnf_probability"] * 100, color=WARN, height=0.6)
    axes[1].set_yticks(y, fc["driver_code"])
    axes[1].set_xlabel("DNF probability (%)")
    axes[1].set_title("Retirement risk")
    _save(fig, "madrid_2026_forecast.png")


def chart_market() -> None:
    from f1pred.evaluation.market import compare_to_market

    fc = pd.read_csv(RESULTS_DIR / FORECAST_RUN / "predictions.csv")
    mk = pd.read_csv(REFERENCE_DIR / "madrid_2026_market_odds.csv")
    c = compare_to_market(fc, mk).head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(c))
    ax.barh(y - 0.2, c["win_probability"] * 100, height=0.38, color=ACCENT, label="model")
    ax.barh(
        y + 0.2,
        c["market_probability"] * 100,
        height=0.38,
        color=MUTED,
        label="market (overround removed)",
    )
    ax.set_yticks(y, c["driver_code"])
    ax.set_xlabel("win probability (%)")
    ax.set_title("Madrid 2026: model versus bookmaker board")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "madrid_model_vs_market.png")


def main() -> None:
    ensure_dirs()
    charts = [
        (
            chart_leaderboard,
            (
                "backtest_tuning",
                "leaderboard_tuning.png",
                "Pre-weekend forecast vs baselines, 2022-2025 (92 races)",
            ),
        ),
        (
            chart_leaderboard,
            (
                "backtest_2026_preweekend",
                "leaderboard_2026.png",
                "Pre-weekend forecast vs baselines, 2026 holdout (13 races)",
            ),
        ),
        (chart_calibration, ()),
        (chart_ablation, ()),
        (chart_circuit_difficulty, ()),
        (chart_madrid, ()),
        (chart_market, ()),
    ]
    for fn, fargs in charts:
        try:
            fn(*fargs)
        except FileNotFoundError as exc:
            print(f"skipped {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
