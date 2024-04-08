from prediction_market_agent_tooling.markets.agent_market import (
    AgentMarket,
    FilterBy,
    SortBy,
)
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket


def get_omen_binary_markets() -> list[OmenAgentMarket]:
    # Get the 5 markets that are closing soonest
    markets: list[AgentMarket] = OmenAgentMarket.get_binary_markets(
        filter_by=FilterBy.OPEN,
        sort_by=SortBy.CLOSING_SOONEST,
        limit=5,
    )


def get_omen_binary_market_from_question(market: str) -> OmenAgentMarket:
    markets = get_omen_binary_markets()
    for m in markets:
        if m.question == market:
            return m
    raise ValueError(f"Market '{market}' not found")


def get_omen_market_token_balance(market: OmenAgentMarket, outcome: bool) -> float:
    # TODO implement this
    return 7.3
