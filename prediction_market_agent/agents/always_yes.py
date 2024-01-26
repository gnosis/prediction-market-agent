from prediction_market_agent.agents.abstract import AbstractAgent
from prediction_market_agent.data_models.market_data_models import AgentMarket


class AlwaysYesAgent(AbstractAgent):
    """
    Save OpenAI API credits. Always say yes!
    """

    def __init__(self):
        pass

    def answer_binary_market(self, market: AgentMarket) -> bool:
        return True
