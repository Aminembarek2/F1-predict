"""Write the entry snapshot a forecast needs, with its availability timestamps.

The forecast refuses to run on an undated entry list, because who is starting is
itself information that arrives at a particular time. This builds the snapshot
from the last completed round and stamps it with the moment those results were
published. Driver or team changes announced since then are the caller's to edit
into the CSV before forecasting.
"""

import argparse
import logging

import pandas as pd

from f1pred.cli import _load_panel
from f1pred.config import CONFIG_DIR, STAGES
from f1pred.forecast import expected_entry_list
from f1pred.leakage import assert_observed_before, prediction_cutoff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--stage", choices=STAGES, default="PRE_WEEKEND")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    panel = _load_panel()
    race_id = f"{args.season}_{args.round:02d}"
    entries = expected_entry_list(panel, args.season, args.round)
    event = panel.schedule.loc[panel.schedule["race_id"].eq(race_id)]
    start = pd.Timestamp(event["race_start_utc"].iloc[0])
    cutoff = prediction_cutoff(start, args.stage)
    assert_observed_before(entries, "observed_at", cutoff)
    entries["cutoff_utc"] = cutoff.isoformat()

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"entries_{race_id}_{args.stage.lower()}.csv"
    entries.to_csv(path, index=False)
    print(entries.to_string(index=False))
    print(f"\nWritten to {path}\ncutoff: {cutoff.isoformat()}")


if __name__ == "__main__":
    main()
