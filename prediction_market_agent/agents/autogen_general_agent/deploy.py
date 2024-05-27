from datetime import datetime, timedelta

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
from prediction_market_agent.agents.microchain_agent.memory import LongTermMemory
from prediction_market_agent.agents.utils import LongTermMemoryTaskIdentifier
from prediction_market_agent.utils import APIKeys


class DeployableSocialMediaAgent(DeployableAgent):
    model: str = "gpt-4-turbo-2024-04-09"
    social_media_handlers: list[AbstractSocialMediaHandler] = []

    def load(self) -> None:
        self.social_media_handlers = [
            FarcasterHandler(),
            TwitterHandler(),
        ]

    def run(self, market_type: MarketType) -> None:
        # It should post a message (cast) on each run.

        one_day_ago = utcnow() - timedelta(days=1)
        bets = self.get_unique_bets_for_market(
            market_type=market_type, start_time=one_day_ago
        )
        # If no bets available for the last 24h, we skip posting.
        if not bets:
            logger.info("No bets available from last day. No post will be created.")
            return

        long_term_memory = LongTermMemory(LongTermMemoryTaskIdentifier.THINK_THOROUGHLY)
        tweet = build_social_media_text(self.model, bets, long_term_memory, one_day_ago)

        # self.post(tweet)

    def get_unique_bets_for_market(
        self, market_type: MarketType, start_time: datetime
    ) -> list[Bet]:
        """
        Returns bets for a given market since start_date.
        Uniqueness defined by market title.
        """
        bets = market_type.market_class.get_bets_made_since(
            better_address=APIKeys().bet_from_address, start_time=start_time
        )
        # filter bets with unique title, i.e. get 1 bet per market
        seen_titles = {bet.market_question: bet for bet in bets}
        filtered_bets = list(seen_titles.values())
        return filtered_bets

    def post(self, tweet: str | None) -> None:
        if not tweet:
            logger.info("No tweet was produced. Exiting.")
            return

        for handler in self.social_media_handlers:
            handler.post(tweet)


if __name__ == "__main__":
    agent = DeployableSocialMediaAgent()
    agent.deploy_local(market_type=MarketType.OMEN, sleep_time=540, timeout=180)
