from unittest.mock import Mock, call, patch

import pytest
from prediction_market_agent_tooling.gtypes import (
    USD,
    OutcomeStr,
    OutcomeToken,
    Probability,
)
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import PlacedTrade, TradeType
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.full_set_collective_arbitrage_agent.deploy import (
    DeployableFullSetCollectiveArbitrageAgent,
)


# --------------------------------------------------------------------------- #
# Helper fixtures                                                             #
# --------------------------------------------------------------------------- #
@pytest.fixture
def mock_market() -> AgentMarket:  # minimal but sufficient market
    m = Mock(spec=AgentMarket)

    m.question = "Mocked market"
    m.outcomes = [OutcomeStr("Yes"), OutcomeStr("No")]
    m.outcome_token_pool = {
        OutcomeStr("Yes"): OutcomeToken(1_000),
        OutcomeStr("No"): OutcomeToken(1_000),
    }
    return m


@pytest.fixture
def agent() -> DeployableFullSetCollectiveArbitrageAgent:
    # Fresh agent instance for every test
    return DeployableFullSetCollectiveArbitrageAgent()


# --------------------------------------------------------------------------- #
# 1. Under-estimated probabilities ⟶ buy + sell                               #
# --------------------------------------------------------------------------- #
@patch.object(DeployableFullSetCollectiveArbitrageAgent, "_max_sets", return_value=5)
@patch.object(DeployableFullSetCollectiveArbitrageAgent, "_trade_complete_sets")
def test_process_market_underestimated(
    trade_complete_sets_mock: Mock,
    max_sets_mock: Mock,
    agent: DeployableFullSetCollectiveArbitrageAgent,
    mock_market: AgentMarket,
) -> None:
    # Probabilities sum to < 1  (⇒ under-estimated)
    mock_market.probabilities = {
        OutcomeStr("Yes"): Probability(0.40),
        OutcomeStr("No"): Probability(0.50),  # 0.90 total
    }

    # Pre-cooked trades the agent is expected to return
    buy_trade = PlacedTrade(
        trade_type=TradeType.BUY, outcome=OutcomeStr("Yes"), amount=USD(10), id="buy-1"
    )
    sell_trade = PlacedTrade(
        trade_type=TradeType.SELL,
        outcome=OutcomeStr("Yes"),
        amount=USD(10),
        id="sell-1",
    )

    # Configure the mock to return different values for different calls
    trade_complete_sets_mock.side_effect = [[buy_trade], [sell_trade]]

    res = agent.process_market(MarketType.SEER, mock_market, verify_market=False)

    assert res is not None, "Arbitrage should be detected"
    assert [buy_trade, sell_trade] == res.trades
    assert res.answer.reasoning is not None
    assert "underestimated" in res.answer.reasoning.lower()

    # Verify the method was called twice with the correct arguments
    expected_calls = [
        call(mock_market, 5, TradeType.BUY),
        call(mock_market, 5, TradeType.SELL),
    ]
    trade_complete_sets_mock.assert_has_calls(expected_calls)


# --------------------------------------------------------------------------- #
# 2. Over-estimated probabilities ⟶ mint + sell                               #
# --------------------------------------------------------------------------- #
@patch.object(DeployableFullSetCollectiveArbitrageAgent, "_arbitrage_overestimated")
def test_process_market_overestimated_large_quantity(
    arbitrage_overestimated_mock: Mock,
    agent: DeployableFullSetCollectiveArbitrageAgent,
    mock_market: AgentMarket,
) -> None:
    # Probabilities sum to > 1  (⇒ over-estimated)
    mock_market.probabilities = {
        OutcomeStr("Yes"): Probability(0.60),
        OutcomeStr("No"): Probability(0.60),  # 1.20 total
    }

    over_trade = PlacedTrade(
        trade_type=TradeType.SELL,
        outcome=OutcomeStr("Yes"),
        amount=USD(12),
        id="over-1",
    )

    # Configure the mock return value
    arbitrage_overestimated_mock.return_value = [over_trade]

    res = agent.process_market(MarketType.SEER, mock_market, verify_market=False)

    assert res is not None
    assert res.trades == [over_trade]
    assert res.answer.reasoning is not None
    assert "overestimated" in res.answer.reasoning.lower()
    arbitrage_overestimated_mock.assert_called_once_with(mock_market)
