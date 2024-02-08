import functions_framework
import random

from prediction_market_agent.data_models.market_data_models import AgentMarket
from prediction_market_agent.deploy.deploy import DeployableAgent
from prediction_market_agent.markets.all_markets import MarketType
from prediction_market_agent.utils import get_keys


class DeployableCoinFlipAgent(DeployableAgent):
    def pick_markets(self, markets: list[AgentMarket]) -> list[AgentMarket]:
        if len(markets) > 1:
            return random.sample(markets, 1)
        return markets

    def answer_binary_market(self, market: AgentMarket) -> bool:
        return random.choice([True, False])


@functions_framework.http
def main(request):
    DeployableCoinFlipAgent().run(market_type=MarketType.MANIFOLD, api_keys=get_keys())
    return "Success"
