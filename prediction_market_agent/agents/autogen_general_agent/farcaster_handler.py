from farcaster import Warpcast
from loguru import logger

from prediction_market_agent.utils import APIKeys


class FarcasterHandler:
    def __init__(self):
        api_keys = APIKeys()
        self.client = Warpcast(
            private_key=api_keys.FARCASTER_PRIVATE_KEY.get_secret_value()
        )

    def post_cast(self, text: str):
        cast = self.client.post_cast(text=text)
        logger.info(f"Posted cast {cast.cast.text} - hash {cast.cast.hash}")
