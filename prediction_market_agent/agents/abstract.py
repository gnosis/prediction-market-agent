import typing as t
from prediction_market_agent.data_models.market_data_models import MarketProtocol

MarketVar = t.TypeVar("MarketVar", bound=MarketProtocol)


class AbstractAgent:
    def pick_market(self, markets: t.Sequence[MarketVar]) -> MarketVar:
        """
        Given a list of markets, pick one to answer.
        """
        # TODO: Pick the market with agent-specific logic, for now just pick the first one.
        return markets[0]

    def answer_binary_market(self, market: MarketProtocol) -> bool:
        """
        Execute the agent, and return the final result as a string.
        """
        raise NotImplementedError
