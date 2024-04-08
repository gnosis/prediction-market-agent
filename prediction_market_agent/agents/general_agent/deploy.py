from dotenv import load_dotenv

load_dotenv()
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.general_agent.general_agent import GeneralAgent


def market_is_saturated(market: AgentMarket) -> bool:
    return market.p_yes > 0.95 or market.p_no > 0.95


class DeployableGeneralAgentAgent(DeployableAgent):
    model = "gpt-3-turbo-preview"
    agent = GeneralAgent()

    def run(self, market_type: MarketType, _place_bet: bool = True) -> None:
        print(f"Agent {self.__class__} starting")
        self.agent.run()
        print(f"Agent {self.__class__} finishing")


if __name__ == "__main__":
    load_dotenv()
    agent = DeployableGeneralAgentAgent()
    agent.deploy_local(
        market_type=MarketType.OMEN, sleep_time=60, timeout=540, place_bet=False
    )
