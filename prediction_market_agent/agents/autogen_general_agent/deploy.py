from datetime import timedelta

from loguru import logger
from prediction_market_agent_tooling.config import PrivateCredentials
from prediction_market_agent_tooling.deploy.agent import (
    DeployableTraderAgent,
)
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


class DeployableSocialMediaAgent(DeployableTraderAgent):
    model: str = "gpt-4-turbo-2024-04-09"
    social_media_handlers: list[AbstractSocialMediaHandler] = [
        FarcasterHandler(),
        TwitterHandler(),
    ]

    def run(self, market_type: MarketType) -> None:
        # It should post a message (cast) on each run.
        # We just need one market to get latest bets.

        bets = self.get_bets(market_type=market_type)
        tweet = build_social_media_text(market_type, bets)
        self.post(tweet)

    def get_bets(self, market_type: MarketType) -> list[Bet]:
        markets = self.get_markets(market_type=market_type, limit=1)
        if not markets:
            raise EnvironmentError(
                f"Could not load market of type {market_type}. Exiting."
            )
        market = markets[0]
        better_address = PrivateCredentials.from_api_keys(APIKeys()).public_key
        one_day_ago = utcnow() - timedelta(days=1)
        return market.get_latest_bets(
            better_address=better_address, start_time=one_day_ago
        )

    def post(self, tweet: str | None):
        if not tweet:
            logger.info("No tweet was produced. Exiting.")
            return

        for handler in self.social_media_handlers:
            handler.post(tweet)


if __name__ == "__main__":
    agent = DeployableSocialMediaAgent()
    agent.deploy_local(market_type=MarketType.OMEN, sleep_time=540, timeout=180)
