from prediction_market_agent_tooling.deploy.agent import Answer, DeployableTraderAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.think_thoroughly_agent.think_thoroughly_agent import (
    CrewAIAgentSubquestions,
)


class DeployableThinkThoroughlyAgent(DeployableTraderAgent):
    model: str = "gpt-4-turbo-2024-04-09"
    bet_on_n_markets_per_run = 1

    def load(self) -> None:
        self.agent = CrewAIAgentSubquestions(model=self.model)

    def answer_binary_market(self, market: AgentMarket) -> Answer | None:
        return self.agent.answer_binary_market(market.question)

    def before(self, market_type: MarketType) -> None:
        self.agent.update_markets()
        super().before(market_type=market_type)


if __name__ == "__main__":
    agent = DeployableThinkThoroughlyAgent(place_bet=False)
    agent.deploy_local(
        market_type=MarketType.OMEN,
        sleep_time=540,
        timeout=180,
    )
