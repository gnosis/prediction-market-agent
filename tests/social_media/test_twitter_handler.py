from typing import Generator

import pytest
from pydantic import SecretStr

from prediction_market_agent.agents.social_media_agent.prompts import POST_MAX_LENGTH
from prediction_market_agent.agents.social_media_agent.social_media.twitter_handler import (
    TwitterHandler,
)
from prediction_market_agent.utils import SocialMediaAPIKeys
from tests.utils import RUN_PAID_TESTS


@pytest.fixture(scope="module")
def twitter_handler_obj() -> Generator[TwitterHandler, None, None]:
    secret_str = SecretStr("test")
    mock_keys = SocialMediaAPIKeys(
        FARCASTER_PRIVATE_KEY=secret_str,
        TWITTER_BEARER_TOKEN=secret_str,
        TWITTER_ACCESS_TOKEN=secret_str,
        TWITTER_ACCESS_TOKEN_SECRET=secret_str,
        TWITTER_API_KEY=secret_str,
        TWITTER_API_KEY_SECRET=secret_str,
    )
    yield TwitterHandler(keys=mock_keys)


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
def test_make_tweet_more_concise(twitter_handler_obj: TwitterHandler) -> None:
    long_tweet = (
        "Account recovery is the most requested use for ZK Email, as losing your wallet hampers crypto "
        "adoption. To solve this, we are open sourcing a generic zk email-based account recovery module for "
        "smart contract wallets with Rhinestone, currently under audit! Here's how it works:  We will give "
        "a general overview of this and how one can get started, stay tuned for more!"
    )

    concise_tweet = twitter_handler_obj.make_tweet_more_concise(long_tweet)
    assert len(concise_tweet) < POST_MAX_LENGTH
