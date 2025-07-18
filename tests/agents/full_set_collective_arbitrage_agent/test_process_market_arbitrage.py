import pytest
from unittest.mock import Mock, patch

from prediction_market_agent_tooling.gtypes       import OutcomeStr, OutcomeToken, Probability, USD
from prediction_market_agent_tooling.markets.data_models import PlacedTrade, TradeType
from prediction_market_agent_tooling.markets.markets      import MarketType
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent.agents.full_set_collective_arbitrage_agent.deploy import (
    DeployableFullSetCollectiveArbitrageAgent,
)


# --------------------------------------------------------------------------- #
# Helper fixtures                                                             #
# --------------------------------------------------------------------------- #
@pytest.fixture
def mock_market() -> AgentMarket:                     # minimal but sufficient market
    m = Mock(spec=AgentMarket)

    m.question  = "Mocked market"
    m.outcomes  = [OutcomeStr("Yes"), OutcomeStr("No")]
    m.outcome_token_pool = {
        OutcomeStr("Yes"): OutcomeToken(1_000),
        OutcomeStr("No"):  OutcomeToken(1_000),
    }
    return m


@pytest.fixture
def agent() -> DeployableFullSetCollectiveArbitrageAgent:
    # Fresh agent instance for every test
    return DeployableFullSetCollectiveArbitrageAgent()


# --------------------------------------------------------------------------- #
# 1. Under-estimated probabilities ⟶ buy + sell                               #
# --------------------------------------------------------------------------- #
@patch.object(DeployableFullSetCollectiveArbitrageAgent, "max_sets_to_buy",  return_value=5)
def test_process_market_underestimated(max_sets_to_buy_mock, agent, mock_market):
    # Probabilities sum to < 1  (⇒ under-estimated)
    mock_market.probabilities = {
        OutcomeStr("Yes"): Probability(0.40),
        OutcomeStr("No"):  Probability(0.50),   # 0.90 total
    }

    # Pre-cooked trades the agent is expected to return
    buy_trade  = PlacedTrade(trade_type=TradeType.BUY,
                            outcome=OutcomeStr("Yes"),
                            amount=USD(10),
                            id="buy-1")
    sell_trade = PlacedTrade(trade_type=TradeType.SELL,
                            outcome=OutcomeStr("Yes"),
                            amount=USD(10),
                            id="sell-1")
    # Stub the heavy on-chain functions
    agent._buy_complete_sets  = Mock(return_value=[buy_trade])
    agent._sell_complete_sets = Mock(return_value=[sell_trade])

    res = agent.process_market(MarketType.SEER, mock_market, verify_market=False)

    assert res is not None, "Arbitrage should be detected"
    assert [buy_trade, sell_trade] == res.trades
    assert "underestimated" in res.answer.reasoning.lower()

    agent._buy_complete_sets.assert_called_once_with(mock_market, 5)
    agent._sell_complete_sets.assert_called_once_with(mock_market, 5)


# --------------------------------------------------------------------------- #
# 2. Over-estimated probabilities ⟶ mint + sell                               #
# --------------------------------------------------------------------------- #
def test_process_market_overestimated(agent, mock_market):
    # Probabilities sum to > 1  (⇒ over-estimated)
    mock_market.probabilities = {
        OutcomeStr("Yes"): Probability(0.60),
        OutcomeStr("No"):  Probability(0.60),   # 1.20 total
    }

    over_trade = PlacedTrade(trade_type=TradeType.SELL,
                            outcome=OutcomeStr("Yes"),
                            amount=USD(12),
                            id="over-1")

    # Short-circuit the inner mint-and-sell logic
    agent._arbitrage_overestimated = Mock(return_value=[over_trade])

    res = agent.process_market(MarketType.SEER, mock_market, verify_market=False)

    assert res is not None
    assert res.trades == [over_trade]
    assert "overestimated" in res.answer.reasoning.lower()
    agent._arbitrage_overestimated.assert_called_once_with(mock_market)



# --------------------------------------------------------------------------- #
# 2. Over-estimated probabilities ⟶ mint + sell  with LARGE quantity                             #
# --------------------------------------------------------------------------- #
def test_process_market_overestimated(agent, mock_market):
    # Probabilities sum to > 1  (⇒ over-estimated)
    mock_market.probabilities = {
        OutcomeStr("Yes"): Probability(0.60),
        OutcomeStr("No"):  Probability(0.60),   # 1.20 total
    }

    over_trade = PlacedTrade(trade_type=TradeType.SELL,
                            outcome=OutcomeStr("Yes"),
                            amount=USD(12),
                            id="over-1")

    # Short-circuit the inner mint-and-sell logic
    agent._arbitrage_overestimated = Mock(return_value=[over_trade])

    res = agent.process_market(MarketType.SEER, mock_market, verify_market=False)

    assert res is not None
    assert res.trades == [over_trade]
    assert "overestimated" in res.answer.reasoning.lower()
    agent._arbitrage_overestimated.assert_called_once_with(mock_market)

