from datetime import datetime
from enum import Enum
from string import Template
from typing import Any, Dict, Optional

import autogen
from autogen import AssistantAgent, UserProxyAgent
from autogen.cache import Cache
from prediction_market_agent_tooling.markets.data_models import Bet
from pydantic import BaseModel

from prediction_market_agent.agents.autogen_general_agent.prompts import (
    CRITIC_PROMPT,
    INFLUENCER_PROMPT,
)
from prediction_market_agent.utils import APIKeys


class BetInputPrompt(BaseModel):
    title: str
    boolean_outcome: bool
    collateral_amount: float
    creation_datetime: datetime

    @staticmethod
    def from_bet(bet: Bet) -> "BetInputPrompt":
        return BetInputPrompt(
            title=bet.market_question,
            boolean_outcome=bet.outcome,
            collateral_amount=bet.amount.amount,
            creation_datetime=bet.created_time,
        )


class AutogenAgentType(str, Enum):
    WRITER = "writer"
    CRITIC = "critic"
    USER = "user"


def reflection_message(
    recipient: UserProxyAgent,
    messages: list[Dict[Any, Any]],
    sender: AssistantAgent,
    config: Optional[Any] = None,
) -> str:
    reflect_prompt = Template(CRITIC_PROMPT).substitute(
        TWEET=recipient.chat_messages_for_summary(sender)[-1]["content"]
    )
    return reflect_prompt


def build_llm_config(model: str) -> Dict[str, Any]:
    keys = APIKeys()
    return {
        "config_list": [
            {"model": model, "api_key": keys.openai_api_key.get_secret_value()}
        ],
    }


def build_agents(model: str) -> Dict[AutogenAgentType, autogen.ConversableAgent]:
    llm_config = build_llm_config(model)

    user_proxy = autogen.UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
        code_execution_config={
            "last_n_messages": 1,
            "work_dir": "tasks",
            "use_docker": False,
        },
    )

    writer = autogen.AssistantAgent(
        name="Writer",
        llm_config=llm_config,
        system_message="""
        You are a professional influencer, known for your insightful and engaging tweets.
        You transform complex concepts into compelling narratives.
        You should improve the quality of the content based on the feedback from the user.
        You must always return only the tweet.
        """,
    )

    critic = autogen.AssistantAgent(
        name="Critic",
        llm_config=llm_config,
        system_message="""
            You are a critic, known for your thoroughness and commitment to standards.
            Your task is to scrutinize content for any harmful elements or regulatory violations, ensuring
            all materials align with required guidelines.
            References to betting and gambling are allowed.
            """,
    )

    return {
        AutogenAgentType.CRITIC: critic,
        AutogenAgentType.WRITER: writer,
        AutogenAgentType.USER: user_proxy,
    }


def build_social_media_text(model: str, bets: list[Bet]) -> str | None:
    """
    Builds a tweet based on the five markets on Omen that are closing soonest.

    This function utilizes the writer and critic agents to generate the tweet content. It first initializes the
    necessary agents based on the provided model. Then, it registers a chat between the writer and critic agents.

    The tweet content is generated using a template that includes questions about each market's title and likelihood.
    """

    agents = build_agents(model)
    user_proxy, writer, critic = (
        agents[AutogenAgentType.USER],
        agents[AutogenAgentType.WRITER],
        agents[AutogenAgentType.CRITIC],
    )
    user_proxy.register_nested_chats(
        [
            {
                "recipient": critic,
                "message": reflection_message,
                "summary_method": "last_msg",
                "max_turns": 1,
            }
        ],
        trigger=writer,
    )

    # ToDO - fetch agent reasoning from DB and construct better tweets
    #  See https://github.com/gnosis/prediction-market-agent/issues/150

    task = Template(INFLUENCER_PROMPT).substitute(
        BETS=[BetInputPrompt.from_bet(bet) for bet in bets]
    )

    # in case we trigger repeated runs, Cache makes it faster.
    with Cache.disk(cache_seed=42) as cache:
        # max_turns = the maximum number of turns for the chat between the two agents. One turn means one conversation round trip.
        res = user_proxy.initiate_chat(
            recipient=writer,
            message=task,
            max_turns=2,
            summary_method="last_msg",
            cache=cache,
        )

    tweet = res.summary
    return str(
        tweet
    )  # Casting needed as summary is of type any and no Pydantic support
