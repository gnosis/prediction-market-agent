from unittest.mock import Mock

import numpy as np
import pytest
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket

from prediction_market_agent.agents.arbitrage_agent.data_models import (
    CorrelatedMarketPair,
    Correlation,
)

PERFECT_POSITIVE_CORRELATION = Correlation(near_perfect_correlation=True, reasoning="")
PERFECT_NEGATIVE_CORRELATION = Correlation(near_perfect_correlation=False, reasoning="")


def build_market(p_yes: float) -> AgentMarket:
    m1 = Mock(OmenAgentMarket, wraps=OmenAgentMarket)
    m1.current_p_yes = p_yes
    m1.current_p_no = 1 - m1.current_p_yes
    return m1


def assert_profit(p1: float, p2: float, correlated_pair: CorrelatedMarketPair) -> None:
    expected_profit = (1 / (p1 + p2)) - 1
    assert np.isclose(
        expected_profit,
        correlated_pair.potential_profit_per_bet_unit(),
        rtol=expected_profit * 0.01,
    )


def assert_bet_amounts_ok(
    correlated_pair: CorrelatedMarketPair,
    main_p: float,
    related_p: float,
) -> None:
    total_bet_amount = 10
    actual_bet = correlated_pair.split_bet_amount_between_yes_and_no(
        total_bet_amount=total_bet_amount
    )
    expected_bet_main = total_bet_amount * main_p / (main_p + related_p)
    expected_bet_related = total_bet_amount * related_p / (main_p + related_p)
    assert np.isclose(
        actual_bet.main_market_bet.size,
        expected_bet_main,
        rtol=expected_bet_main * 0.01,
    )
    assert np.isclose(
        actual_bet.related_market_bet.size,
        expected_bet_related,
        rtol=expected_bet_related * 0.01,
    )


@pytest.mark.parametrize(
    "p1, p2, corr, m1_bet_yes, m2_bet_yes",
    [
        (0.5, 0.8, PERFECT_POSITIVE_CORRELATION, True, False),  # YES/NO
        (0.8, 0.5, PERFECT_POSITIVE_CORRELATION, False, True),  # NO/YES
        (0.2, 0.5, PERFECT_NEGATIVE_CORRELATION, True, True),  # YES/YES
        (0.8, 0.5, PERFECT_NEGATIVE_CORRELATION, False, False),  # NO/NO
    ],
)
def test_profit(
    p1: float, p2: float, corr: Correlation, m1_bet_yes: bool, m2_bet_yes: bool
) -> None:
    m1 = build_market(p1)
    m2 = build_market(p2)
    correlated_pair = CorrelatedMarketPair(
        main_market=m1, related_market=m2, correlation=corr
    )
    p_market1 = m1.current_p_yes if m1_bet_yes else m1.current_p_no
    p_market2 = m2.current_p_yes if m2_bet_yes else m2.current_p_no

    assert_profit(p_market1, p_market2, correlated_pair)
    assert_bet_amounts_ok(correlated_pair, p_market1, p_market2)

    bet_direction_main, bet_direction_related = correlated_pair.bet_directions()

    assert bet_direction_main == m1_bet_yes
    assert bet_direction_related == m2_bet_yes
