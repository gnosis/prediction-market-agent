import asyncio
from enum import Enum
from string import Template

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import BaseGroupChat, MagenticOneGroupChat
from autogen_agentchat.teams._group_chat._magentic_one._prompts import (
    ORCHESTRATOR_FINAL_ANSWER_PROMPT,
)
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from prediction_market_agent_tooling.gtypes import CollateralToken
from prediction_market_agent_tooling.markets.data_models import Bet
from prediction_market_agent_tooling.tools.langfuse_ import observe
from prediction_market_agent_tooling.tools.utils import DatetimeUTC
from pydantic import BaseModel

from prediction_market_agent.agents.microchain_agent.memory import (
    SimpleMemoryThinkThoroughly,
)
from prediction_market_agent.agents.social_media_agent.prompts import (
    INFLUENCER_PROMPT,
    POST_MAX_LENGTH,
    REASONING_PROMPT,
)
from prediction_market_agent.agents.utils import extract_reasonings_to_learnings
from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemoryTableHandler,
)
from prediction_market_agent.utils import APIKeys


# Options from https://microsoft.github.io/autogen/docs/reference/agentchat/conversable_agent/#initiate_chat
class SummaryMethod(str, Enum):
    LAST_MSG = "last_msg"
    REFLECTION_WITH_LLM = "reflection_with_llm"


class BetInputPrompt(BaseModel):
    title: str
    boolean_outcome: bool
    collateral_amount: CollateralToken
    creation_datetime: DatetimeUTC

    @staticmethod
    def from_bet(bet: Bet) -> "BetInputPrompt":
        return BetInputPrompt(
            title=bet.market_question,
            boolean_outcome=bet.outcome,
            collateral_amount=bet.amount,
            creation_datetime=bet.created_time,
        )


class AutogenAgentType(str, Enum):
    WRITER = "writer"
    CRITIC = "critic"
    USER = "user"


def build_team(model: str) -> BaseGroupChat:
    model_client = OpenAIChatCompletionClient(
        model=model, api_key=APIKeys().openai_api_key.get_secret_value()
    )

    primary_agent = AssistantAgent(
        "primary",
        model_client=model_client,
        description="Generate a tweet for the user. Make sure the feedback given by the critic is being addressed.",
        system_message="""You are a professional influencer, known for your insightful and engaging tweets. You transform complex concepts into compelling narratives. You should improve the quality of the content based on the feedback from the user. You must always return only the tweet.""",
    )

    critic_agent = AssistantAgent(
        "critic",
        model_client=model_client,
        description="Critique the tweet and provide feedback.",
        system_message=f""" You are a critic, known for your thoroughness and commitment to standards. Your task is to scrutinize content for any harmful elements or regulatory violations, ensuring all materials align with required guidelines. You should also always remind everyone that the limit for any posts being created is {POST_MAX_LENGTH} characters. References to betting and gambling are allowed.""",
    )

    final_prompt = (
        ORCHESTRATOR_FINAL_ANSWER_PROMPT
        + "\n Output only the final version of the tweet and nothing else."
    )

    magentic_team = MagenticOneGroupChat(
        [primary_agent, critic_agent],
        model_client=model_client,
        final_answer_prompt=final_prompt,
    )
    return magentic_team


@observe()
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
    long_term_memory: LongTermMemoryTableHandler,
    memories_since: DatetimeUTC | None = None,
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
        m
        for m in simple_memories
        if m.metadata.original_question in questions_from_bets
    ]
    return extract_reasonings_to_learnings(filtered_memories, tweet)


@observe()
def build_reply_tweet(
    model: str,
    tweet: str,
    bets: list[Bet],
    long_term_memory: LongTermMemoryTableHandler,
    memories_since: DatetimeUTC | None = None,
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
    team = build_team(model)

    task_result = asyncio.run(Console(team.run_stream(task=task)))
    reply_tweet = task_result.messages[-1].content  # Last message is critic's approval
    return str(reply_tweet)
