from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from prediction_market_agent_tooling.loggers import logger
from tweepy import Client

from prediction_market_agent.agents.social_media_agent.prompts import POST_MAX_LENGTH
from prediction_market_agent.agents.social_media_agent.social_media.abstract_handler import (
    AbstractSocialMediaHandler,
)
from prediction_market_agent.utils import APIKeys, SocialMediaAPIKeys


class TwitterHandler(AbstractSocialMediaHandler):
    client: Client
    llm: BaseChatModel

    def __init__(self, model: str = "gpt-4", keys: SocialMediaAPIKeys = None) -> None:
        if keys is None:
            keys = SocialMediaAPIKeys()
        self.client = Client(
            keys.twitter_bearer_token.get_secret_value(),
            keys.twitter_api_key.get_secret_value(),
            keys.twitter_api_key_secret.get_secret_value(),
            keys.twitter_access_token.get_secret_value(),
            keys.twitter_access_token_secret.get_secret_value(),
        )

        self.llm = ChatOpenAI(
            temperature=0,
            model=model,
            api_key=APIKeys().openai_api_key.get_secret_value(),
        )

    def make_tweet_more_concise(self, tweet: str) -> str:
        system_template = f"Make this tweet more concise while keeping an analytical tone. You are forbidden of using more than {POST_MAX_LENGTH} characters."
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_template), ("user", "{text}")]
        )

        chain = prompt_template | self.llm | StrOutputParser()
        result = chain.invoke({"text": tweet})
        return result

    @staticmethod
    def does_post_length_exceed_max_length(tweet: str) -> bool:
        return len(tweet) > POST_MAX_LENGTH

    def post(self, text: str, reasoning_reply_tweet: str) -> None:
        quote_tweet_id = self.post_else_retry_with_summarization(text)
        self.post_else_retry_with_summarization(
            reasoning_reply_tweet, quote_tweet_id=quote_tweet_id
        )

    def post_else_retry_with_summarization(
        self, text: str, quote_tweet_id: str | None = None
    ) -> str | None:
        """
        Posts the provided text on Twitter and retries with a summarized version if the text exceeds the maximum
        length. If the summarized text also exceeds the maximum length, a warning is logged and the function returns
        without posting.
        """
        if self.does_post_length_exceed_max_length(text):
            if self.does_post_length_exceed_max_length(
                text := self.make_tweet_more_concise(text)
            ):
                logger.warning(
                    f"Tweet too long. Length: {len(text)}, max length: {POST_MAX_LENGTH}"
                )
                return None
        posted_tweet = self.client.create_tweet(
            text=text, quote_tweet_id=quote_tweet_id
        )
        logger.debug(f"Tweeted {text} - {posted_tweet}")
        try:
            return str(posted_tweet.data["id"])
        except Exception as e:
            logger.debug("Could not extract tweet ID. Exception: ", e)
            return None
