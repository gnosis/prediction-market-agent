from microchain import Function
from prediction_market_agent_tooling.loggers import logger

from prediction_market_agent.agents.microchain_agent.microchain_agent_keys import (
    MicrochainAgentKeys,
)
from prediction_market_agent.agents.social_media_agent.social_media.twitter_handler import (
    POST_MAX_LENGTH,
    TwitterHandler,
)


class SendTweet(Function):
    @property
    def description(self) -> str:
        return f"Use this function to post a tweet on Twitter. Maximum length of the tweet is {POST_MAX_LENGTH} characters."

    @property
    def example_args(self) -> list[str]:
        return ["This is my tweet."]

    def __call__(self, tweet: str) -> str:
        if TwitterHandler.does_post_length_exceed_max_length(tweet):
            return f"Tweet length exceeds the maximum allowed length of {POST_MAX_LENGTH} characters, because it is {len(tweet)} characters long. Please shorten the tweet."
        if MicrochainAgentKeys().ENABLE_SOCIAL_MEDIA:
            # Use the raw `post_tweet`, instead of `post`, let the general agent shorten the tweet on its own if it's required.
            TwitterHandler().post_tweet(tweet)
        else:
            # Log as error, so we are notified about it, if we forget to turn it on in production.
            logger.error(f"Social media is disabled. Tweeting skipped: {tweet}")
        return "Tweet sent."


TWITTER_FUNCTIONS: list[type[Function]] = [
    SendTweet,
]
