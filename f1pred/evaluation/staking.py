"""Does the forecast's calibration convert into money, or is it already priced?

Calibration is a property of a forecast; edge is a property of a forecast *and a
price*. A forecast can be perfectly calibrated and still worthless to act on,
because the market already knew. This module is the test of the second thing.

Two rules here are easy to get wrong and change the answer completely:

1. **Stakes are priced at the quoted odds, never at the de-vigged ones.** The
   overround is removed to compare probabilities; it must be left in to compare
   money, because the bookmaker pays the quoted price. De-vigging before staking
   invents an edge that nobody can collect.
2. **A book with an overround of ``v`` costs the average bettor ``1 - 1/v``.**
   At the 22.7% overround on the reference board that is 18.5% of turnover, so a
   forecast has to disagree with the market by a great deal before any of it is
   recoverable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .market import american_to_implied


def american_to_decimal(odds: pd.Series) -> pd.Series:
    """Convert American moneyline odds to decimal (total return per unit staked)."""
    o = pd.to_numeric(odds, errors="coerce")
    return (1.0 + o / 100.0).where(o > 0, 1.0 + 100.0 / o.abs())


def kelly_fraction(probability: pd.Series, decimal_odds: pd.Series) -> pd.Series:
    """Kelly stake as a fraction of bankroll; zero where the bet is not positive.

    ``f* = (p*b - q) / b`` with ``b`` the net return per unit staked. Negative
    values mean "lay it", which an outright win market does not allow, so they
    are floored at zero rather than inverted.
    """
    p = pd.to_numeric(probability, errors="coerce")
    b = pd.to_numeric(decimal_odds, errors="coerce") - 1.0
    edge = p * b - (1.0 - p)
    return (edge / b).where(b > 0).clip(lower=0.0).fillna(0.0)


def price_board(
    forecast: pd.DataFrame,
    odds: pd.DataFrame,
    *,
    probability_col: str = "win_probability",
    odds_col: str = "american_odds",
    key: str = "driver_code",
) -> pd.DataFrame:
    """Join a forecast to a quoted board and price every selection on it."""
    board = odds.copy()
    board["decimal_odds"] = american_to_decimal(board[odds_col])
    board["implied_raw"] = american_to_implied(board[odds_col])
    merged = forecast[[key, probability_col]].merge(board, on=key, how="inner", validate="1:1")
    merged = merged.rename(columns={probability_col: "model_probability"})
    overround = float(board["implied_raw"].sum())
    merged["overround"] = overround
    merged["market_probability"] = merged["implied_raw"] / overround
    # Expected return per unit staked at the quoted price. Zero is break-even.
    merged["expected_value"] = merged["model_probability"] * merged["decimal_odds"] - 1.0
    merged["edge_vs_market_pp"] = 100.0 * (
        merged["model_probability"] - merged["market_probability"]
    )
    merged["kelly_fraction"] = kelly_fraction(merged["model_probability"], merged["decimal_odds"])
    return merged.sort_values("expected_value", ascending=False).reset_index(drop=True)


def stake_plan(
    board: pd.DataFrame,
    *,
    bankroll: float = 1000.0,
    kelly_multiplier: float = 0.25,
    min_expected_value: float = 0.05,
) -> pd.DataFrame:
    """Select and size the bets a board actually justifies.

    Fractional Kelly, because full Kelly assumes the probabilities are correct and
    these are estimates with their own error. The expected-value floor exists for
    the same reason: an edge of one percent is inside the model's own uncertainty
    and is not a signal, it is noise that happens to be positive.
    """
    if not 0 < kelly_multiplier <= 1:
        raise ValueError("kelly_multiplier must be in (0, 1]")
    if min_expected_value < 0:
        raise ValueError("an expected-value floor below zero would stake losing bets")
    plan = board.loc[board["expected_value"] >= min_expected_value].copy()
    plan["stake"] = bankroll * kelly_multiplier * plan["kelly_fraction"]
    return plan.loc[plan["stake"] > 0].reset_index(drop=True)


def settle(plan: pd.DataFrame, winner_key: str, *, key: str = "driver_code") -> pd.DataFrame:
    """Settle a staked plan against the race result."""
    out = plan.copy()
    out["won"] = (out[key] == winner_key).astype(int)
    out["returned"] = np.where(out["won"].eq(1), out["stake"] * out["decimal_odds"], 0.0)
    out["profit"] = out["returned"] - out["stake"]
    return out


def staking_report(settled: pd.DataFrame) -> dict[str, float]:
    """Summarise a settled plan. On a handful of bets this is mostly variance."""
    staked = float(settled["stake"].sum())
    profit = float(settled["profit"].sum())
    return {
        "bets": int(len(settled)),
        "staked": staked,
        "profit": profit,
        "roi": profit / staked if staked > 0 else float("nan"),
        "expected_roi": float((settled["expected_value"] * settled["stake"]).sum() / staked)
        if staked > 0
        else float("nan"),
        "wins": int(settled["won"].sum()),
    }


def market_efficiency(board: pd.DataFrame) -> dict[str, float]:
    """How much room the board leaves, before any question of who is right.

    Reported whether or not a single bet is placed: if the model's disagreement
    with the market is smaller than the bookmaker's margin, no staking rule can
    turn that forecast into money, and the honest conclusion is that the forecast
    is already priced.
    """
    overround = float(board["overround"].iloc[0])
    disagreement = board["edge_vs_market_pp"].abs()
    return {
        "overround": overround,
        "house_margin_pct": 100.0 * (1.0 - 1.0 / overround),
        "mean_abs_disagreement_pp": float(disagreement.mean()),
        "max_abs_disagreement_pp": float(disagreement.max()),
        "selections_with_positive_ev": int((board["expected_value"] > 0).sum()),
        "selections": int(len(board)),
        "best_expected_value": float(board["expected_value"].max()),
    }
