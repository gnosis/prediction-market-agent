from prediction_market_agent_tooling.markets.data_models import AgentMarket

from prediction_market_agent.agents.abstract import AbstractAgent


class AlwaysYesAgent(AbstractAgent):
    """
    Save OpenAI API credits. Always say yes!
    """

    def __init__(self) -> None:
        pass

    def answer_binary_market(self, market: AgentMarket) -> bool:
        return True
