import typing as t

from crewai_tools import BaseTool
from prediction_market_agent_tooling.markets.agent_market import AgentMarket, FilterBy
from prediction_market_agent_tooling.markets.markets import (
    MarketType,
    get_binary_markets,
)


# ToDo - Add filters as arguments to this function call, add it to description.
class GetBinaryMarketsTool(BaseTool):
    name: str = "Tool for fetching binary markets"
    description: str = (
        "This tool returns all the markets on the prediction market platform Omen."
    )

    def _run(self) -> list[AgentMarket]:
        # Implementation goes here
        markets = get_binary_markets(
            limit=10,
            market_type=MarketType.OMEN,
            filter_by=FilterBy.NONE,
        )
        return markets

    def cache_function(self, args: list[t.Any], result: t.Any) -> bool:
        # ToDo - implement caching if result was > 1 and depending on filter.
        return False
