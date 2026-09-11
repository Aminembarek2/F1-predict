"""Download the overtakes and pit stops that the forecaster currently guesses at.

Overtaking difficulty is inferred today from how much the finishing order
resembles the grid, and strategy variance is a hand-chosen constant. A timing
feed reports both directly, so both can be measured instead.
"""

import argparse
import logging

from f1pred.cli import _load_panel
from f1pred.config import RAW_DIR, RESULTS_DIR, ensure_dirs
from f1pred.features.circuits import measured_overtaking_rate, pit_lane_loss
from f1pred.sources.openf1 import load_race_events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    ensure_dirs()

    panel = _load_panel()
    overtakes, pit, coverage = load_race_events(args.seasons, panel.schedule, panel.results)
    coverage.to_csv(RESULTS_DIR / "race_events_coverage.csv", index=False)
    if overtakes.empty:
        raise SystemExit("no overtake data returned; nothing was written")

    overtakes.to_parquet(RAW_DIR / "overtakes.parquet", index=False)
    if not pit.empty:
        pit.to_parquet(RAW_DIR / "pit.parquet", index=False)

    rate = measured_overtaking_rate(overtakes, panel.results)
    loss = pit_lane_loss(pit)
    rate.to_csv(RESULTS_DIR / "overtaking_measured.csv", index=False)
    loss.to_csv(RESULTS_DIR / "pit_lane_loss.csv", index=False)
    print(f"{overtakes['race_id'].nunique()} races, {len(overtakes)} passes, {len(pit)} stops")
    print()
    print(rate.round(2).to_string(index=False))
    print()
    print(loss.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
