from prediction_market_agent.agents.abstract import AbstractAgent
from prediction_market_agent.data_models.market_data_models import MarketProtocol


class AlwaysYesAgent(AbstractAgent):
    """
    Save OpenAI API credits. Always say yes!
    """

    def __init__(self) -> None:
        pass

    def answer_boolean_market(self, market: MarketProtocol) -> bool:
        return True
