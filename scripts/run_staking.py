"""Price a forecast against a quoted board and report whether it leaves an edge.

This answers the question calibration cannot: the forecast is well calibrated,
but is it saying anything the market has not already priced? Run it on one race
and the profit figure is noise; run it on a corpus of boards captured before each
race and it becomes the only evaluation that matters for acting on the forecast.
"""

import argparse
import json
import logging

import pandas as pd

from f1pred.config import RESULTS_DIR, ensure_dirs
from f1pred.evaluation.staking import (
    market_efficiency,
    price_board,
    settle,
    stake_plan,
    staking_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", required=True, help="results/<run>/predictions.csv")
    parser.add_argument("--odds", required=True, help="board CSV with driver_code, american_odds")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--kelly", type=float, default=0.25)
    parser.add_argument("--min-ev", type=float, default=0.05)
    parser.add_argument("--winner", help="driver_code of the actual winner, once it is known")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    ensure_dirs()

    forecast = pd.read_csv(args.forecast)
    board = price_board(forecast, pd.read_csv(args.odds))
    efficiency = market_efficiency(board)
    plan = stake_plan(
        board,
        bankroll=args.bankroll,
        kelly_multiplier=args.kelly,
        min_expected_value=args.min_ev,
    )

    print(
        board[
            [
                "driver_code",
                "model_probability",
                "market_probability",
                "decimal_odds",
                "edge_vs_market_pp",
                "expected_value",
                "kelly_fraction",
            ]
        ]
        .head(12)
        .round(4)
        .to_string(index=False)
    )
    print()
    print(json.dumps(efficiency, indent=2))

    report = None
    if not plan.empty:
        print()
        print(
            plan[["driver_code", "decimal_odds", "expected_value", "stake"]]
            .round(2)
            .to_string(index=False)
        )
        if args.winner:
            settled = settle(plan, args.winner)
            report = staking_report(settled)
            print()
            print(json.dumps(report, indent=2))
    else:
        print("\nNo selection clears the expected-value floor: nothing to stake.")

    out = RESULTS_DIR / "staking"
    out.mkdir(parents=True, exist_ok=True)
    board.to_csv(out / "board.csv", index=False)
    plan.to_csv(out / "plan.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "forecast": args.forecast,
                "odds": args.odds,
                "kelly_multiplier": args.kelly,
                "min_expected_value": args.min_ev,
                "market_efficiency": efficiency,
                "settlement": report,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
