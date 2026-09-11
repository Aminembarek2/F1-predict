"""Build the historical entry and grid snapshots a strict backtest needs.

Without these the backtest can only read the field off the finishing results of
the race it is predicting, which is how it learns who was withdrawn. Each
snapshot here is constructed from information that existed before its cutoff:

* the entry list is the previous round's starters, stamped with the moment that
  round's results were published;
* the grid, used only from POST_QUALI, is the starting order of the target race,
  stamped at the qualifying session rather than at the race.

A season opener has no previous round to copy, so it gets no snapshot and is
dropped from the strict backtest rather than guessed at.
"""

import argparse
import logging

import pandas as pd

from f1pred.cli import _load_panel
from f1pred.config import CONFIG_DIR, STAGE_ORDER, STAGES, ensure_dirs
from f1pred.leakage import prediction_cutoff

log = logging.getLogger(__name__)

#: Qualifying runs the afternoon before the race on a standard weekend. Stamping
#: the grid earlier than the true session end would overstate its availability.
GRID_OFFSET_HOURS = -24.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    parser.add_argument("--stage", choices=STAGES, default="PRE_WEEKEND")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    ensure_dirs()

    panel = _load_panel()
    results = panel.results
    wants_grid = STAGE_ORDER[args.stage] >= STAGE_ORDER["POST_QUALI"]

    rows, dropped = [], []
    targets = (
        panel.schedule.loc[panel.schedule["season"].isin(args.seasons)]
        .merge(panel.event_order[["race_id", "event_index"]], on="race_id")
        .sort_values("event_index")
    )
    for event in targets.to_dict("records"):
        race_id = event["race_id"]
        if results.loc[results["race_id"].eq(race_id)].empty:
            continue
        history = results.loc[results["event_index"] < event["event_index"]]
        previous = history.loc[history["event_index"].eq(history["event_index"].max())]
        if previous.empty or int(previous["season"].iloc[0]) != int(event["season"]):
            dropped.append({"race_id": race_id, "reason": "season_opener_has_no_prior_round"})
            continue

        start = pd.Timestamp(event["race_start_utc"])
        cutoff = prediction_cutoff(start, args.stage)
        entries = (
            previous.loc[previous["started"].eq(1), ["driver_id", "driver_code", "constructor_id"]]
            .drop_duplicates("driver_id")
            .sort_values("driver_id")
            .reset_index(drop=True)
        )
        entries["season"] = int(event["season"])
        entries["round"] = int(event["round"])
        entries["race_id"] = race_id
        entries["circuit_id"] = event["circuit_id"]
        entries["source"] = "previous_event_projection"
        entries["observed_at"] = previous["historical_available_at"].max()
        entries["cutoff_utc"] = cutoff.isoformat()

        if wants_grid:
            grid = results.loc[results["race_id"].eq(race_id)].set_index("driver_id")["grid"]
            entries["grid"] = entries["driver_id"].map(grid)
            # A named driver who never took a grid slot starts from the back of
            # the field the snapshot itself declared.
            entries["grid"] = entries["grid"].replace(0, pd.NA).fillna(len(entries))
            entries["grid_observed_at"] = (
                start + pd.to_timedelta(GRID_OFFSET_HOURS, unit="h")
            ).isoformat()
        rows.append(entries)

    if not rows:
        raise SystemExit("no snapshots could be built")
    frame = pd.concat(rows, ignore_index=True)
    path = (
        CONFIG_DIR / f"snapshots_{min(args.seasons)}_{max(args.seasons)}_{args.stage.lower()}.csv"
    )
    frame.to_csv(path, index=False)
    print(f"{frame['race_id'].nunique()} races, {len(frame)} entry rows -> {path}")
    for row in dropped:
        print(f"  dropped {row['race_id']}: {row['reason']}")


if __name__ == "__main__":
    main()
