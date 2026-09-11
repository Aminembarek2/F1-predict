"""Convenience wrapper for the shared backtest CLI."""

import sys

from f1pred.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["backtest", *sys.argv[1:]]))
