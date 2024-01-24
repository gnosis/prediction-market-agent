import typing as t
from prediction_market_agent.data_models.market_data_models import MarketProtocol

MarketVar = t.TypeVar("MarketVar", bound=MarketProtocol)


class AbstractAgent:
    def __init__(self):
        pass

    def pick_market(self, markets: list[MarketVar]) -> MarketVar:
        """
        Given a list of markets, pick one to answer.
        """
        raise NotImplementedError

    def answer_boolean_market(self, market: MarketProtocol) -> bool:
        """
        Execute the agent, and return the final result as a string.
        """
        raise NotImplementedError
