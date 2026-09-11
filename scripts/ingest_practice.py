"""Download and cache the practice laps used by the experiment runner.

Two public sources are supported. OpenF1 is tried first and returns 401 for every
endpoint, historical ones included, while any session is live; the F1 timing
archive read through FastF1 is the fallback. Whichever source answers, the lap
contract handed to the experiment is identical.
"""

import argparse
import logging

import pandas as pd

from f1pred.cli import _load_panel
from f1pred.config import RAW_DIR, RESULTS_DIR, ensure_dirs
from f1pred.sources import openf1, timing

log = logging.getLogger(__name__)
SOURCES = {"openf1": openf1.load_practice, "timing": timing.load_practice}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    parser.add_argument("--source", choices=["auto", *SOURCES], default="auto")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    ensure_dirs()
    panel = _load_panel()

    order = list(SOURCES) if args.source == "auto" else [args.source]
    laps, coverage, used = pd.DataFrame(), pd.DataFrame(), None
    for name in order:
        try:
            laps, coverage = SOURCES[name](args.seasons, panel.schedule, panel.results)
        except Exception as exc:  # noqa: BLE001 - the fallback exists for any source outage
            log.warning("practice source %s unavailable: %s", name, exc)
            continue
        if not laps.empty:
            used = name
            break
    if used is None:
        raise SystemExit("no practice source returned laps; nothing was written")

    # A source may stop early on its own rate limit. Merging with what is already
    # on disk lets a second run resume instead of starting the calendar again.
    coverage["source"] = used
    path = RAW_DIR / "practice.parquet"
    if path.exists():
        kept = pd.read_parquet(path)
        kept = kept.loc[~kept["race_id"].isin(set(laps["race_id"]))]
        laps = pd.concat([kept, laps], ignore_index=True)
    laps = laps.sort_values(["race_id", "observed_at"]).reset_index(drop=True)
    laps.to_parquet(path, index=False)
    coverage.to_csv(RESULTS_DIR / "practice_coverage.csv", index=False)
    print(f"source: {used}")
    print(coverage.to_string(index=False))
    print(f"weekends on disk: {laps['race_id'].nunique()}, laps: {len(laps)}")


if __name__ == "__main__":
    main()
