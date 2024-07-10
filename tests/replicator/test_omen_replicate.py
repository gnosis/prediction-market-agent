from prediction_market_agent_tooling.markets.manifold.api import get_manifold_market
from prediction_market_agent_tooling.markets.manifold.manifold import (
    ManifoldAgentMarket,
)

from prediction_market_agent.agents.replicate_to_omen_agent.omen_replicate import (
    filter_out_markets_with_banned_categories,
)


def test_filter_out_markets_with_banned_categories() -> None:
    markets = [
        ManifoldAgentMarket.from_data_model(
            get_manifold_market("5cjxwtve8d")
        ),  # Banned
        ManifoldAgentMarket.from_data_model(
            get_manifold_market("1sfbjcTu6zgqjCVjqWA2")
        ),  # Okay
    ]
    filtered = filter_out_markets_with_banned_categories(markets)
    assert len(filtered) == 1, "Should have filtered out the banned market."
    assert filtered[0] == markets[1], "Should have kept the okay market."
