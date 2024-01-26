from prediction_market_agent.data_models.market_data_models import AgentMarket


class AbstractAgent:
    def __init__(self):
        pass

    def pick_market(self, markets: list[AgentMarket]) -> AgentMarket:
        """
        Given a list of markets, pick one to answer.
        """
        # TODO: Pick the market with agent-specific logic, for now just pick the first one.
        return markets[0]

    def answer_binary_market(self, market: AgentMarket) -> bool:
        """
        Execute the agent, and return the final result as a string.
        """
        raise NotImplementedError
