from farcaster import Warpcast
from farcaster.models import Parent
from prediction_market_agent_tooling.loggers import logger

from prediction_market_agent.agents.autogen_general_agent.social_media.abstract_handler import (
    AbstractSocialMediaHandler,
)
from prediction_market_agent.utils import SocialMediaAPIKeys


class FarcasterHandler(AbstractSocialMediaHandler):
    def __init__(self) -> None:
        api_keys = SocialMediaAPIKeys()
        self.client = Warpcast(
            private_key=api_keys.farcaster_private_key.get_secret_value()
        )

    def post(self, text: str, reasoning_reply_tweet: str) -> None:
        cast = self.client.post_cast(text=text)
        logger.info(f"Posted cast {cast.cast.text} - hash {cast.cast.hash}")
        parent_cast = Parent(fid=1, hash=cast.cast.hash)
        reply_cast = self.client.post_cast(reasoning_reply_tweet, parent=parent_cast)
        logger.info(
            f"Posted reply cast {reply_cast.cast.text} - hash {reply_cast.cast.hash}"
        )
