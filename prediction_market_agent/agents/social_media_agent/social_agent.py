import asyncio
from enum import Enum
from string import Template
from typing import Any, Dict, Optional

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import BaseGroupChat, RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from prediction_market_agent_tooling.markets.data_models import Bet
from prediction_market_agent_tooling.tools.langfuse_ import observe
from prediction_market_agent_tooling.tools.utils import DatetimeUTC
from pydantic import BaseModel

from prediction_market_agent.agents.microchain_agent.memory import (
    SimpleMemoryThinkThoroughly,
)
from prediction_market_agent.agents.social_media_agent.prompts import (
    CRITIC_PROMPT,
    INFLUENCER_PROMPT,
    REASONING_PROMPT,
    POST_MAX_LENGTH,
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
    collateral_amount: float
    creation_datetime: DatetimeUTC

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


def build_team(model: str) -> BaseGroupChat:
    # user_proxy = UserProxyAgent(
    #     name="User",
    #     human_input_mode="NEVER",
    #     is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    #     code_execution_config={
    #         "last_n_messages": 1,
    #         "work_dir": "tasks",
    #         "use_docker": False,
    #     },
    # )

    # writer = AssistantAgent(
    #     name="Writer",
    #     llm_config=llm_config,
    #     system_message="""You are a professional influencer, known for your insightful and engaging tweets. You
    #     transform complex concepts into compelling narratives. You should improve the quality of the content based on
    #     the feedback from the user. You must always return only the tweet. """,
    # )

    #     critic = AssistantAgent(
    #         name="Critic",
    #         llm_config=llm_config,
    #         system_message=f""" You are a critic, known for your thoroughness and commitment to standards. Your task is
    #         to scrutinize content for any harmful elements or regulatory violations, ensuring all materials align with
    #         required guidelines. You should also always remind everyone that the limit for any posts being created is
    # {POST_MAX_LENGTH} characters. References to betting and gambling are allowed.""",
    #     )

    # Create an OpenAI model client.
    model_client = OpenAIChatCompletionClient(
        model=model, api_key=APIKeys().openai_api_key.get_secret_value()
    )

    # Create the primary agent.
    primary_agent = AssistantAgent(
        "primary",
        model_client=model_client,
        description="Generate a tweet for the user. Make sure the feedback given by the critic is being addressed.",
        system_message="""You are a professional influencer, known for your insightful and engaging tweets. You transform complex concepts into compelling narratives. You should improve the quality of the content based on the feedback from the user. You must always return only the tweet.""",
    )

    # Create the critic agent.
    critic_agent = AssistantAgent(
        "critic",
        model_client=model_client,
        description="Critique the tweet and provide feedback.",
        system_message=f""" You are a critic, known for your thoroughness and commitment to standards. Your task is to scrutinize content for any harmful elements or regulatory violations, ensuring all materials align with required guidelines. You should also always remind everyone that the limit for any posts being created is {POST_MAX_LENGTH} characters. References to betting and gambling are allowed.""",
        # Respond with 'APPROVE' to when your feedbacks are addressed.""",
    )

    text_termination = TextMentionTermination("APPROVE")
    max_message_termination = MaxMessageTermination(5)
    # Combine the termination conditions using the `|`` operator so that the
    # task stops when either condition is met.
    termination = text_termination | max_message_termination

    reflection_team = RoundRobinGroupChat(
        [primary_agent, critic_agent], termination_condition=termination, max_turns=4
    )
    return reflection_team


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

    # ToDo - prompt better so it always produces a tweet
    # ToDo - Follow this https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/tutorial/teams.html#team-usage-guide

    task_result = asyncio.run(Console(team.run_stream(task=task)))
    reply_tweet = task_result.messages[-2].content  # Last message is critic's approval

    # in case we trigger repeated runs, Cache makes it faster.
    # with Cache.disk(cache_seed=42) as cache:
    #     # max_turns = the maximum number of turns for the chat between the two agents. One turn means one conversation round trip.
    #     res = user_proxy.initiate_chat(
    #         recipient=writer,
    #         message=task,
    #         max_turns=2,
    #         summary_method=SummaryMethod.REFLECTION_WITH_LLM,
    #         cache=cache,
    #     )
    # We extract the last message since the revised tweet is contained in the last response
    # from the writer.
    # reply_tweet = res.chat_history[-1]["content"]

    return str(
        reply_tweet
    )  # Casting needed as summary is of type any and no Pydantic support
