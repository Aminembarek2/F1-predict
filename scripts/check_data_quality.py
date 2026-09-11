"""Measure the two data-quality defects the audit turned up, season by season.

Both shaped the design, so both are re-measured from the current panel rather
than asserted from memory: retirement causes stopped being reported in 2024, and
a "Lapped" status is a classified finish rather than a failure.
"""

import argparse
import logging

import pandas as pd

from f1pred.cli import _load_panel
from f1pred.config import RESULTS_DIR, ensure_dirs

GENERIC = {"Retired", "Did not finish", ""}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", type=int, default=2014)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    ensure_dirs()

    results = _load_panel().results
    results = results.loc[results["season"] >= args.since]
    rows = []
    for season, group in results.groupby("season"):
        retired = group.loc[group["dnf"].eq(1)]
        status = retired["status"].fillna("").astype(str)
        lapped = group["status"].fillna("").astype(str).str.contains("Lapped", case=False)
        rows.append(
            {
                "season": int(season),
                "entries": len(group),
                "retirements": len(retired),
                "dnf_rate": float(group["dnf"].mean()),
                "distinct_causes": int(status.nunique()),
                "generic_cause_share": float(status.isin(GENERIC).mean()) if len(status) else 0.0,
                # Counting a lapped car as a retirement is what produced the
                # implausible 50% DNF rate the audit found.
                "lapped_rows": int(lapped.sum()),
                "dnf_rate_if_lapped_counted": float((group["dnf"].eq(1) | lapped).mean()),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "data_quality.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
