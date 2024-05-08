from loguru import logger
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.autogen_general_agent.social_agent import (
    build_social_media_text,
)
from prediction_market_agent.agents.autogen_general_agent.social_media.abstract_handler import (
    AbstractSocialMediaHandler,
)
from prediction_market_agent.agents.autogen_general_agent.social_media.farcaster_handler import (
    FarcasterHandler,
)
from prediction_market_agent.agents.autogen_general_agent.social_media.twitter_handler import (
    TwitterHandler,
)


class DeployableFarcasterAgent(DeployableAgent):
    model: str = "gpt-4-turbo-2024-04-09"
    social_media_handlers: list[AbstractSocialMediaHandler] = [
        FarcasterHandler(),
        TwitterHandler(),
    ]

    def run(self, market_type: MarketType) -> None:
        # It should post a message (cast) on each run.
        tweet = build_social_media_text(self.model)
        if tweet:
            for handler in self.social_media_handlers:
                handler.post(tweet)

        else:
            logger.info("Post could not be constructed, exiting.")


if __name__ == "__main__":
    agent = DeployableFarcasterAgent()
    agent.deploy_local(market_type=MarketType.OMEN, sleep_time=540, timeout=180)
