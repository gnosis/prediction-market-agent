import pytest

from prediction_market_agent.agents.social_media_agent.prompts import POST_MAX_LENGTH
from prediction_market_agent.agents.social_media_agent.social_media.twitter_handler import (
    TwitterHandler,
)
from prediction_market_agent.utils import SocialMediaAPIKeys


@pytest.fixture(scope="module")
def twitter_handler_obj() -> TwitterHandler:
    mock_keys = SocialMediaAPIKeys(
        FARCASTER_PRIVATE_KEY="test",
        TWITTER_BEARER_TOKEN="test",
        TWITTER_ACCESS_TOKEN="test",
        TWITTER_ACCESS_TOKEN_SECRET="test",
        TWITTER_API_KEY="test",
        TWITTER_API_KEY_SECRET="test",
    )
    yield TwitterHandler(keys=mock_keys)


def test_dummy(twitter_handler_obj: TwitterHandler):
    long_tweet = (
        "Account recovery is the most requested use for ZK Email, as losing your wallet hampers crypto "
        "adoption. To solve this, we are open sourcing a generic zk email-based account recovery module for "
        "smart contract wallets with Rhinestone, currently under audit! Here's how it works:  We will give "
        "a general overview of this and how one can get started, stay tuned for more!"
    )

    concise_tweet = twitter_handler_obj.make_tweet_more_concise(long_tweet)
    assert len(concise_tweet) < POST_MAX_LENGTH
