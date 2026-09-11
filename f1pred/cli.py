"""F1 Predictor: cached ingestion, explicit forecast snapshots and evaluation."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .artifacts import write_manifest
from .config import RAW_DIR, RESULTS_DIR, SEED, STAGES, ensure_dirs
from .pipeline import Panel, build_panel


def _load_panel(race_score: str = "rank") -> Panel:
    missing = [
        n for n in ("results", "qualifying", "schedule") if not (RAW_DIR / f"{n}.parquet").exists()
    ]
    if missing:
        raise SystemExit(f"missing raw data {missing}; run `f1pred ingest` first")
    standings = RAW_DIR / "standings.parquet"
    overtakes = RAW_DIR / "overtakes.parquet"
    return build_panel(
        pd.read_parquet(RAW_DIR / "results.parquet"),
        pd.read_parquet(RAW_DIR / "qualifying.parquet"),
        pd.read_parquet(RAW_DIR / "schedule.parquet"),
        pd.read_parquet(standings) if standings.exists() else None,
        pd.read_parquet(overtakes) if overtakes.exists() else None,
        race_score=race_score,
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    from .sources import jolpica

    ensure_dirs()
    for name, loader in (
        ("schedule", jolpica.load_schedule),
        ("results", jolpica.load_results),
        ("qualifying", jolpica.load_qualifying),
    ):
        frame = loader(args.seasons, refresh=args.refresh)
        frame.to_parquet(RAW_DIR / f"{name}.parquet", index=False)
        logging.info("wrote %s rows to %s.parquet", len(frame), name)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from .evaluation.backtest import leaderboard, walk_forward
    from .evaluation.metrics import paired_table, probability_report

    ensure_dirs()
    panel = _load_panel()
    ids = (
        panel.results.loc[panel.results["season"].isin(args.seasons)]
        .sort_values("event_index")["race_id"]
        .unique()
        .tolist()
    )
    snapshots, cutoffs = None, None
    if args.snapshots:
        frame = pd.read_csv(args.snapshots)
        snapshots = {str(rid): group for rid, group in frame.groupby("race_id")}
        cutoffs = {rid: str(group["cutoff_utc"].iloc[0]) for rid, group in snapshots.items()}
    baselines = (
        ("championship", "last_race", "form5", "uniform")
        if not panel.standings.empty
        else ("gp_points", "last_race", "form5", "uniform")
    )
    predictions, per_race = walk_forward(
        panel,
        ids,
        stage=args.stage,
        n_sims=args.sims,
        baselines=baselines,
        allow_retrospective=args.allow_retrospective,
        snapshots=snapshots,
        cutoffs=cutoffs,
        include_grid_reference=args.grid_reference,
    )
    if predictions.empty:
        raise ValueError("no races could be evaluated")
    output = RESULTS_DIR / args.out
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    per_race.to_csv(output / "per_race.csv", index=False)
    table = leaderboard(per_race)
    table.to_csv(output / "leaderboard.csv", index=False)
    # A leaderboard on its own invites a point-estimate reading, so the paired
    # intervals and the calibration of every published probability are written
    # beside it, from the same run.
    paired = paired_table(per_race, incumbent="strength_mc")
    paired.to_csv(output / "paired_tests.csv", index=False)
    report = probability_report(predictions.loc[predictions["model"].eq("strength_mc")])
    report.to_csv(output / "probability_report.csv", index=False)
    write_manifest(
        output / "manifest.json",
        command="f1pred backtest",
        stage=args.stage,
        seasons=args.seasons,
        n_sims=args.sims,
        seed=SEED,
        evaluation_scope=str(predictions["evaluation_scope"].iloc[0]),
        calibration=report.to_dict("records"),
    )
    print(table.round(4).to_string(index=False))
    print()
    print(report.round(4).to_string(index=False))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .forecast import forecast_upcoming

    ensure_dirs()
    entries = pd.read_csv(args.entries)
    result = forecast_upcoming(
        _load_panel(),
        season=args.season,
        round_no=args.round,
        stage=args.stage,
        n_sims=args.sims,
        entries=entries,
        cutoff=args.cutoff,
        allow_retrospective=args.allow_retrospective,
    )
    output = RESULTS_DIR / f"forecast_{args.season}_{args.round:02d}_{args.stage.lower()}"
    output.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(output / "predictions.csv", index=False)
    write_manifest(
        output / "manifest.json",
        command="f1pred predict",
        n_sims=args.sims,
        seed=SEED,
        stage=args.stage,
        cutoff=args.cutoff,
        circuit=asdict(result.circuit),
        diagnostics=result.diagnostics,
        evaluation_scope=result.table["evaluation_scope"].iloc[0],
    )
    columns = [
        "predicted_rank",
        "driver_id",
        "win_probability",
        "podium_probability",
        "top10_probability",
        "dnf_probability",
        "expected_finish",
    ]
    print(result.table[columns].round(4).to_string(index=False))
    print(f"Written to {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="f1pred", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="cache the historical calendar, results and qualifying")
    ingest.add_argument("--seasons", type=int, nargs="+", default=list(range(2014, 2027)))
    ingest.add_argument("--refresh", action="store_true")
    ingest.set_defaults(func=cmd_ingest)
    backtest = sub.add_parser("backtest", help="evaluate explicit historical snapshots")
    backtest.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    backtest.add_argument(
        "--snapshots", type=Path, help="entry/grid CSV with race_id and cutoff_utc"
    )
    backtest.add_argument("--grid-reference", action="store_true")
    backtest.add_argument("--out", default="backtest")
    backtest.set_defaults(func=cmd_backtest)
    predict = sub.add_parser("predict", help="forecast from an explicit entry/grid snapshot")
    predict.add_argument("--season", type=int, required=True)
    predict.add_argument("--round", type=int, required=True)
    predict.add_argument("--entries", type=Path, required=True)
    predict.add_argument("--cutoff", required=True, help="UTC snapshot cutoff, in ISO 8601 format")
    predict.set_defaults(func=cmd_predict)
    for command in (backtest, predict):
        command.add_argument("--stage", choices=STAGES, default="PRE_WEEKEND")
        command.add_argument("--sims", type=int, default=20000)
        command.add_argument(
            "--allow-retrospective",
            action="store_true",
            help="explicit conditional diagnostic; ineligible for model adoption",
        )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if hasattr(args, "out") and (Path(args.out).name != args.out or args.out in {".", ".."}):
        parser.error("--out must be a directory name within results")
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
