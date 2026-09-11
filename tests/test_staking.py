"""The staking layer decides money, so its two easy mistakes are pinned here."""

import numpy as np
import pandas as pd
import pytest

from f1pred.evaluation.staking import (
    american_to_decimal,
    kelly_fraction,
    market_efficiency,
    price_board,
    settle,
    stake_plan,
    staking_report,
)


def _board():
    forecast = pd.DataFrame(
        {"driver_code": ["AAA", "BBB", "CCC"], "win_probability": [0.50, 0.30, 0.20]}
    )
    # Implied 0.5 / 0.3333 / 0.25 = 1.0833 overround.
    odds = pd.DataFrame({"driver_code": ["AAA", "BBB", "CCC"], "american_odds": [100, 200, 300]})
    return price_board(forecast, odds)


def test_american_to_decimal_handles_both_signs():
    out = american_to_decimal(pd.Series([100, -200, 300]))
    np.testing.assert_allclose(out.to_numpy(), [2.0, 1.5, 4.0])


def test_kelly_is_zero_unless_the_bet_is_positive():
    # p=0.5 at decimal 2.0 is exactly break-even, so it stakes nothing.
    assert kelly_fraction(pd.Series([0.5]), pd.Series([2.0])).iloc[0] == pytest.approx(0.0)
    # p=0.6 at decimal 2.0: f* = (0.6*1 - 0.4)/1 = 0.2
    assert kelly_fraction(pd.Series([0.6]), pd.Series([2.0])).iloc[0] == pytest.approx(0.2)
    assert kelly_fraction(pd.Series([0.3]), pd.Series([2.0])).iloc[0] == pytest.approx(0.0)


def test_expected_value_is_priced_at_the_quoted_odds_not_the_fair_ones():
    """De-vigging before staking invents an edge nobody can collect.

    The bookmaker pays the quoted price. Removing the overround first makes the
    prices better than the ones on offer, which manufactures positive expected
    value out of the house margin itself.
    """
    board = _board().set_index("driver_code")
    row = board.loc["AAA"]
    # Quoted +100 is decimal 2.0, so a 0.50 forecast is exactly break-even.
    assert row["expected_value"] == pytest.approx(0.0)
    # De-vigged, the same selection prices at 1/0.4615 = 2.167 and looks like a
    # +8.3% edge. That edge is the house margin, handed back on paper only.
    fair_decimal = 1.0 / row["market_probability"]
    assert row["model_probability"] * fair_decimal - 1.0 == pytest.approx(0.0833, abs=1e-3)
    # BBB is genuinely negative at the quoted price: 0.30 * 3.0 - 1 = -0.10.
    assert board.loc["BBB", "expected_value"] == pytest.approx(-0.10)


def test_stake_plan_respects_the_expected_value_floor():
    board = _board()
    everything = stake_plan(board, min_expected_value=0.0)
    picky = stake_plan(board, min_expected_value=0.20)
    assert len(picky) <= len(everything)
    assert (picky["expected_value"] >= 0.20).all()
    with pytest.raises(ValueError, match="below zero"):
        stake_plan(board, min_expected_value=-0.1)


def test_settlement_pays_the_quoted_price_and_loses_the_rest():
    plan = pd.DataFrame(
        {
            "driver_code": ["AAA", "BBB"],
            "decimal_odds": [2.0, 4.0],
            "expected_value": [0.1, 0.1],
            "stake": [100.0, 50.0],
        }
    )
    settled = settle(plan, "BBB")
    assert settled.set_index("driver_code").loc["AAA", "profit"] == pytest.approx(-100.0)
    assert settled.set_index("driver_code").loc["BBB", "profit"] == pytest.approx(150.0)
    report = staking_report(settled)
    assert report["bets"] == 2 and report["wins"] == 1
    assert report["profit"] == pytest.approx(50.0)
    assert report["roi"] == pytest.approx(50.0 / 150.0)


def test_market_efficiency_reports_the_house_margin():
    summary = market_efficiency(_board())
    assert summary["overround"] == pytest.approx(1.08333, abs=1e-4)
    # A 1.0833 book keeps 1 - 1/1.0833 = 7.7% of turnover.
    assert summary["house_margin_pct"] == pytest.approx(7.6923, abs=1e-3)
    assert summary["selections"] == 3
