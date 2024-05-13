import tweepy
from prediction_market_agent_tooling.loggers import logger
from tweepy import Client

from prediction_market_agent.agents.autogen_general_agent.social_media.abstract_handler import (
    AbstractSocialMediaHandler,
)
from prediction_market_agent.utils import SocialMediaAPIKeys


class TwitterHandler(AbstractSocialMediaHandler):
    client: Client

    def __init__(self) -> None:
        keys = SocialMediaAPIKeys()

        self.client = tweepy.Client(
            keys.twitter_bearer_token.get_secret_value(),
            keys.twitter_api_key.get_secret_value(),
            keys.twitter_api_key_secret.get_secret_value(),
            keys.twitter_access_token.get_secret_value(),
            keys.twitter_access_token_secret.get_secret_value(),
        )

    def post(self, text: str) -> None:
        response = self.client.create_tweet(text=text)
        logger.info(f"Posted tweet {text} - response {response}")
