import pytest
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.tools.prediction_market import GetMarkets


# TODO investigate why this fails for polymarket https://github.com/gnosis/prediction-market-agent/issues/62
@pytest.mark.parametrize("market_type", [MarketType.OMEN, MarketType.MANIFOLD])
def test_get_markets(market_type: MarketType) -> None:
    get_markets = GetMarkets(market_type=market_type)
    assert len(get_markets._run()) > 0
