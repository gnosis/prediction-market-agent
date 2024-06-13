from datetime import datetime
from enum import Enum
from string import Template
from typing import Any, Dict, Optional

import autogen
from autogen import AssistantAgent, UserProxyAgent
from autogen.cache import Cache
from prediction_market_agent_tooling.markets.data_models import Bet
from pydantic import BaseModel

from prediction_market_agent.agents.social_media_agent.prompts import (
    CRITIC_PROMPT,
    INFLUENCER_PROMPT,
    REASONING_PROMPT,
)
from prediction_market_agent.agents.microchain_agent.memory import (
    LongTermMemory,
    SimpleMemoryThinkThoroughly,
)
from prediction_market_agent.agents.utils import extract_reasonings_to_learnings
from prediction_market_agent.utils import APIKeys


# Options from https://microsoft.github.io/autogen/docs/reference/agentchat/conversable_agent/#initiate_chat
class SummaryMethod(str, Enum):
    LAST_MSG = "last_msg"
    REFLECTION_WITH_LLM = "reflection_with_llm"


POST_MAX_LENGTH = 280


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
        system_message="""You are a professional influencer, known for your insightful and engaging tweets. You 
        transform complex concepts into compelling narratives. You should improve the quality of the content based on 
        the feedback from the user. You must always return only the tweet. """,
    )

    critic = autogen.AssistantAgent(
        name="Critic",
        llm_config=llm_config,
        system_message=f""" You are a critic, known for your thoroughness and commitment to standards. Your task is 
        to scrutinize content for any harmful elements or regulatory violations, ensuring all materials align with 
        required guidelines. You should also always remind everyone that the limit for any posts being created is 
{POST_MAX_LENGTH} characters. References to betting and gambling are allowed.""",
    )

    return {
        AutogenAgentType.CRITIC: critic,
        AutogenAgentType.WRITER: writer,
        AutogenAgentType.USER: user_proxy,
    }


def build_social_media_text(
    model: str,
    bets: list[Bet],
) -> str:
    """
    Builds a tweet based on past betting activity from a given participant.

    This function utilizes the writer and critic agents to generate the tweet content. It first initializes the
    necessary agents based on the provided model. Then, it registers a chat between the writer and critic agents.

    The tweet content is generated using a template that includes questions about each market's title and likelihood.
    """

    task = Template(INFLUENCER_PROMPT).substitute(
        BETS=[BetInputPrompt.from_bet(bet) for bet in bets],
    )

    tweet = build_tweet(model=model, task=task)
    return tweet


def extract_reasoning_behind_tweet(
    tweet: str,
    bets: list[Bet],
    long_term_memory: LongTermMemory,
    memories_since: datetime | None = None,
) -> str:
    """
    Fetches memories from the DB that are most closely related to bets.
    Returns a summary of the reasoning value from the metadata of those memories.
    """
    memories = long_term_memory.search(from_=memories_since)
    simple_memories = [
        SimpleMemoryThinkThoroughly.from_long_term_memory(ltm) for ltm in memories
    ]
    # We want memories only from the bets to add relevant learnings
    questions_from_bets = set([b.market_question for b in bets])
    filtered_memories = [
        m for m in simple_memories if m.metadata.question in questions_from_bets
    ]
    return extract_reasonings_to_learnings(filtered_memories, tweet)


def build_reply_tweet(
    model: str,
    tweet: str,
    bets: list[Bet],
    long_term_memory: LongTermMemory,
    memories_since: datetime | None = None,
) -> str:
    reasoning = extract_reasoning_behind_tweet(
        tweet=tweet,
        bets=bets,
        long_term_memory=long_term_memory,
        memories_since=memories_since,
    )

    task = Template(REASONING_PROMPT).substitute(
        TWEET=tweet,
        REASONING=reasoning,
    )

    tweet = build_tweet(model=model, task=task)
    return tweet


def build_tweet(model: str, task: str) -> str:
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

    # in case we trigger repeated runs, Cache makes it faster.
    with Cache.disk(cache_seed=42) as cache:
        # max_turns = the maximum number of turns for the chat between the two agents. One turn means one conversation round trip.
        res = user_proxy.initiate_chat(
            recipient=writer,
            message=task,
            max_turns=2,
            summary_method=SummaryMethod.REFLECTION_WITH_LLM,
            cache=cache,
        )
    # We extract the last message since the revised tweet is contained in the last response
    # from the writer.
    reply_tweet = res.chat_history[-1]["content"]

    return str(
        reply_tweet
    )  # Casting needed as summary is of type any and no Pydantic support
