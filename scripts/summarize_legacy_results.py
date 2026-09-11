from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from f1pred.legacy.proxy_experiment import run_uploaded_results_summary  # noqa: E402


def wilson(k, n, z=1.959963984540054):
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def find_file(name, explicit=None):
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    for p in [
        ROOT / "data" / "reference" / name,
        ROOT.parent / name,
        Path("/mnt/data") / name,
        Path.cwd() / name,
    ]:
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout")
    ap.add_argument("--feature-experiments")
    ap.add_argument("--stress2026")
    a = ap.parse_args()
    res = ROOT / "results" / "archive" / "original"
    img = ROOT / "images" / "archive" / "original"
    ref = ROOT / "data" / "reference"
    for p in [res, img, ref]:
        p.mkdir(parents=True, exist_ok=True)

    h = find_file("holdout_predictions.csv", a.holdout)
    f = find_file("feature_experiments.csv", a.feature_experiments)
    s = find_file("2026_regime_stress_test.csv", a.stress2026)
    for p in [h, f, s]:
        if p:
            dst = ref / p.name
            if p.resolve() != dst.resolve():
                shutil.copy2(p, dst)
    # Preserve the other archived outputs when present.
    for name in [
        "feature_catalog.csv",
        "feature_importance_gain.csv",
        "final_ranker_metrics.csv",
        "model_leaderboard.csv",
    ]:
        p = find_file(name)
        if p:
            dst = ref / name
            if p.resolve() != dst.resolve():
                shutil.copy2(p, dst)

    if not h or not f:
        raise SystemExit("Need holdout_predictions.csv and feature_experiments.csv")

    empirical = run_uploaded_results_summary(str(h), str(f))
    empirical.to_csv(res / "proxy_ablation_2025.csv", index=False)

    # Uncertainty from aggregate hit counts; paired uncertainty cannot be recovered without per-race predictions from each ablation model.
    rows = []
    for _, r in empirical.iterrows():
        if pd.isna(r.get("top1")):
            continue
        n = int(r.get("n_races", 24))
        k = int(round(float(r.top1) * n))
        lo, hi = wilson(k, n)
        rows.append(
            {
                "feature_set": r.feature_set,
                "top1": r.top1,
                "wins": k,
                "n_races": n,
                "wilson95_low": lo,
                "wilson95_high": hi,
            }
        )
    unc = pd.DataFrame(rows)
    unc.to_csv(res / "proxy_uncertainty_2025.csv", index=False)

    if s:
        stress = pd.read_csv(s)
        stress.to_csv(res / "regime_stress_test_2026.csv", index=False)

    # Archived per-race output, preserved for error analysis.
    hh = pd.read_csv(h)
    hh.to_csv(res / "holdout_predictions_2025.csv", index=False)
    by = []
    hh["grid_score"] = -pd.to_numeric(hh.grid_effective, errors="coerce").fillna(99)
    for rid, g in hh.groupby("raceId"):
        actual = g.loc[g.winner == 1].iloc[0]
        grid = g.loc[g.grid_score.idxmax()]
        ranker = g.loc[g.ranker_score.idxmax()] if "ranker_score" in g else grid
        by.append(
            {
                "raceId": rid,
                "race_name": actual.get("race_name", rid),
                "actual_winner": actual.get("code", actual.driverId),
                "grid_pick": grid.get("code", grid.driverId),
                "grid_correct": int(grid.winner == 1),
                "final_ranker_pick": ranker.get("code", ranker.driverId),
                "final_ranker_correct": int(ranker.winner == 1),
            }
        )
    pd.DataFrame(by).to_csv(res / "proxy_vs_grid_by_race_2025.csv", index=False)

    # NOTE: this script previously wrote results/module_status.csv and
    # results/validation_tests_expanded.csv as hard-coded PASS rows, whether or
    # not pytest had run. That fabricated a test report, so those writers have
    # been removed. Test evidence now comes only from an actual pytest run:
    #     pytest -q --junitxml=results/pytest_junit.xml
    # See docs/AUDIT.md, finding 3.

    # Charts
    plot = empirical.dropna(subset=["top1"]).copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(plot.feature_set, plot.top1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Top-1 winner accuracy")
    ax.set_title("2025 empirical feature-family ablation (uploaded run)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(img / "proxy_ablation_top1.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(plot.feature_set, plot.ndcg5)
    ax.set_ylim(0.75, 1)
    ax.set_ylabel("NDCG@5")
    ax.set_title("2025 ranking quality by feature family")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(img / "proxy_ablation_ndcg5.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    uu = unc.dropna()
    fig, ax = plt.subplots(figsize=(12, 5))
    y = uu.top1.to_numpy()
    lo = y - uu.wilson95_low.to_numpy()
    hi = uu.wilson95_high.to_numpy() - y
    ax.errorbar(np.arange(len(uu)), y, yerr=np.vstack([lo, hi]), fmt="o", capsize=4)
    ax.set_xticks(np.arange(len(uu)), uu.feature_set, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Top-1 accuracy (Wilson 95% CI)")
    ax.set_title("2025 winner-accuracy uncertainty")
    fig.tight_layout()
    fig.savefig(img / "proxy_top1_uncertainty.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "empirical_best_2025": plot.sort_values("top1", ascending=False).iloc[0].to_dict(),
        "grid_baseline_2025": plot[plot.feature_set.str.contains("GRID_BASELINE_NO_ML", na=False)]
        .iloc[0]
        .to_dict()
        if (plot.feature_set == "GRID_BASELINE_NO_ML").any()
        else {},
        "evidence_boundary": "The corner-geometry features are implementation-tested only; any predictive lift would have to be measured on a real telemetry backtest.",
    }
    summary["note"] = (
        "Reproduces the archived summary only. Test counts are not asserted here; run pytest for test evidence."
    )
    (res / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
