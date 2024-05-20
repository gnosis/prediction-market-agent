from prediction_market_agent_tooling.deploy.agent import Answer, DeployableTraderAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.think_thoroughly_agent.think_thoroughly_agent import (
    CrewAIAgentSubquestions,
)
from prediction_market_agent.utils import APIKeys


class DeployableThinkThoroughlyAgent(DeployableTraderAgent):
    model: str = "gpt-4-turbo-2024-04-09"
    bet_on_n_markets_per_run = 1

    def load(self) -> None:
        self.agent = CrewAIAgentSubquestions(model=self.model)

    def answer_binary_market(self, market: AgentMarket) -> Answer | None:
        return self.agent.answer_binary_market(market.question)


if __name__ == "__main__":
    agent = DeployableThinkThoroughlyAgent(place_bet=False)
    # from crewai.utilities import paths
    # db_path = f"{paths.db_storage_path()}/long_term_memory_storage.db"
    # print(db_path)
    # ToDo delete me
    import os

    k = APIKeys()
    os.environ["OPENAI_API_KEY"] = k.openai_api_key.get_secret_value()
    agent.deploy_local(
        market_type=MarketType.OMEN,
        sleep_time=540,
        timeout=180,
    )
