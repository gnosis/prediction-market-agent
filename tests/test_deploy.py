import random

from prediction_market_agent.data_models.market_data_models import AgentMarket
from prediction_market_agent.deploy.deploy import DeployableAgent, DeploymentType
from prediction_market_agent.markets.all_markets import MarketType
from prediction_market_agent.utils import get_keys


def test_local_deployment():
    class DeployableCoinFlipAgent(DeployableAgent):
        def pick_markets(self, markets: list[AgentMarket]) -> list[AgentMarket]:
            if len(markets) > 1:
                return random.sample(markets, 1)
            return markets

        def answer_binary_market(self, market: AgentMarket) -> bool:
            return random.choice([True, False])

    agent = DeployableCoinFlipAgent()
    agent.deploy(
        sleep_time=0.001,
        market_type=MarketType.MANIFOLD,
        deployment_type=DeploymentType.LOCAL,
        api_keys=get_keys(),
        timeout=0.01,
        place_bet=False,
    )
