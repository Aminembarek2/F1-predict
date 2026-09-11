"""Cache official standings, including sprint points and official tie breaks."""

import argparse
import logging

import pandas as pd

from f1pred.cli import _load_panel
from f1pred.config import RAW_DIR
from f1pred.sources.jolpica import load_driver_standings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    panel = _load_panel()
    events = panel.results.loc[
        panel.results["season"].isin(args.seasons), ["season", "round"]
    ].drop_duplicates()
    frames = []
    for season, rnd in events.itertuples(index=False, name=None):
        logging.info("standings after %s round %s", season, rnd)
        frames.append(load_driver_standings(int(season), int(rnd)))
    if not frames:
        return
    standings = pd.concat(frames, ignore_index=True)
    # Seasons already on disk are kept, so one season can be refreshed alone.
    path = RAW_DIR / "standings.parquet"
    if path.exists():
        kept = pd.read_parquet(path)
        kept = kept.loc[~kept["season"].isin(set(standings["season"]))]
        standings = pd.concat([kept, standings], ignore_index=True)
    standings = standings.sort_values(["season", "after_round", "championship_position"])
    standings.reset_index(drop=True).to_parquet(path, index=False)
    print(standings.groupby("season")["after_round"].nunique().to_string())


if __name__ == "__main__":
    main()
