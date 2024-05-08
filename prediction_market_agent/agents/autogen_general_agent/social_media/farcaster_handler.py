from farcaster import Warpcast
from loguru import logger

from prediction_market_agent.agents.autogen_general_agent.social_media.abstract_handler import (
    AbstractSocialMediaHandler,
)
from prediction_market_agent.utils import APIKeys, SocialMediaAPIKeys


class FarcasterHandler(AbstractSocialMediaHandler):
    def __init__(self) -> None:
        api_keys = SocialMediaAPIKeys()
        self.client = Warpcast(
            private_key=api_keys.farcaster_private_key.get_secret_value()
        )

    def post(self, text: str) -> None:
        cast = self.client.post_cast(text=text)
        logger.info(f"Posted cast {cast.cast.text} - hash {cast.cast.hash}")
