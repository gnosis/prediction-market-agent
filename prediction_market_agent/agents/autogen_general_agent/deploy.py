from datetime import timedelta

from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.data_models import Bet
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.utils import utcnow

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
from prediction_market_agent.utils import APIKeys


class DeployableSocialMediaAgent(DeployableAgent):
    model: str = "gpt-4-turbo-2024-04-09"
    social_media_handlers: list[AbstractSocialMediaHandler] = [
        FarcasterHandler(),
        TwitterHandler(),
    ]

    def run(self, market_type: MarketType) -> None:
        # It should post a message (cast) on each run.

        bets = self.get_bets(market_type=market_type)
        # If no bets available for the last 24h, we skip posting.
        if not bets:
            logger.info("No bets available from last day. No post will be created.")
            return
        tweet = build_social_media_text(self.model, bets)
        self.post(tweet)

    def get_bets(self, market_type: MarketType) -> list[Bet]:
        one_day_ago = utcnow() - timedelta(days=1)
        return market_type.market_class.get_bets_made_since(
            better_address=APIKeys().bet_from_address, start_time=one_day_ago
        )

    def post(self, tweet: str | None) -> None:
        if not tweet:
            logger.info("No tweet was produced. Exiting.")
            return

        for handler in self.social_media_handlers:
            handler.post(tweet)


if __name__ == "__main__":
    agent = DeployableSocialMediaAgent()
    agent.deploy_local(market_type=MarketType.OMEN, sleep_time=540, timeout=180)
