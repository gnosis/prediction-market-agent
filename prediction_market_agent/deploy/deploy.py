from enum import Enum
import subprocess
import time
import tempfile
from pydantic import BaseModel

from prediction_market_agent.data_models.market_data_models import AgentMarket
from prediction_market_agent.markets.all_markets import (
    MarketType,
    get_binary_markets,
    place_bet,
)
from prediction_market_agent.tools.betting_strategies import get_tiny_bet
from prediction_market_agent.utils import APIKeys


class DeploymentType(str, Enum):
    GOOGLE_CLOUD = "google_cloud"
    LOCAL = "local"


class DeployableAgent(BaseModel):
    def pick_markets(self, markets: list[AgentMarket]) -> list[AgentMarket]:
        if len(markets) > 1:
            return markets[:1]
        return markets

    def answer_binary_market(self, market: AgentMarket) -> bool:
        raise NotImplementedError("This method should be implemented by the subclass")

    def deploy(
        self,
        sleep_time: int,
        market_type: MarketType,
        deployment_type: DeploymentType,
        api_keys: APIKeys,
    ):
        if deployment_type == DeploymentType.GOOGLE_CLOUD:
            # Deploy to Google Cloud Functions, and use Google Cloud Scheduler to run the function
            gcp_function_name = self.get_gcloud_fname(market_type=MarketType.MANIFOLD)

            cmd = (
                f"gcloud functions deploy {gcp_function_name} "
                f"--runtime=python310 "
                f"--trigger-http "
                f"--allow-unauthenticated "
                f"--entry-point= "
                # f"--source={file} "
            )
            # subprocess.run(cmd, shell=True)
        elif deployment_type == DeploymentType.LOCAL:
            while True:
                self.run(market_type, api_keys)
                time.sleep(sleep_time)

    def run(self, market_type: MarketType, api_keys: APIKeys):
        available_markets = [
            x.to_agent_market() for x in get_binary_markets(market_type)
        ]
        markets = self.pick_markets(available_markets)
        for market in markets:
            result = self.answer_binary_market(market)
            print(f"Placing bet on {market} with result {result}")
            place_bet(
                market=market.original_market,
                amount=get_tiny_bet(market_type),
                outcome=result,
                keys=api_keys,
                omen_auto_deposit=True,
            )

    def get_gcloud_fname(self, market_type: MarketType) -> str:
        return f"{self.__class__.__name__.lower()}-{market_type}-{int(time.time())}"
