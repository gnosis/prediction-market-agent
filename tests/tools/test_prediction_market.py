import pytest
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.tools.prediction_market import (
    GetMarketProbability,
    GetMarkets,
)


# TODO investigate why this fails for polymarket https://github.com/gnosis/prediction-market-agent/issues/62
@pytest.mark.parametrize("market_type", [MarketType.OMEN, MarketType.MANIFOLD])
def test_get_markets(market_type: MarketType) -> None:
    get_markets = GetMarkets(market_type=market_type)
    assert len(get_markets._run()) > 0


@pytest.mark.parametrize("market_type", [MarketType.OMEN])
def test_get_probability(market_type: MarketType) -> None:
    market_id = "0x0020d13c89140b47e10db54cbd53852b90bc1391"
    get_market_probability = GetMarketProbability(market_type=market_type)
    assert float(get_market_probability(market_id)[0]) == 0.0
    market: AgentMarket = market_type.market_class.get_binary_market(market_id)
    assert market.is_resolved()  # Probability wont change after resolution
