from loguru import logger
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.autogen_general_agent.farcaster_agent import (
    build_tweet,
)
from prediction_market_agent.agents.autogen_general_agent.farcaster_handler import (
    FarcasterHandler,
)


class DeployableFarcasterAgent(DeployableAgent):
    model: str = "gpt-4-turbo-2024-04-09"

    def load(self) -> None:
        self.farcaster_handler = FarcasterHandler()

    def run(self, market_type: MarketType, _place_bet: bool = True) -> None:
        # It should post a message (cast) on each run.
        tweet = build_tweet(self.model)
        if tweet:
            self.farcaster_handler.post_cast(tweet)
        else:
            logger.info("Post could not be constructed, exiting.")


if __name__ == "__main__":
    agent = DeployableFarcasterAgent()
    agent.deploy_local(market_type=MarketType.OMEN, sleep_time=540, timeout=180)
