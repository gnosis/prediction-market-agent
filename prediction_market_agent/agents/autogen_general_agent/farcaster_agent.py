import datetime
from enum import Enum
from string import Template
from typing import Dict, Optional, Any

import autogen
from autogen import AssistantAgent, UserProxyAgent
from autogen.cache import Cache
from loguru import logger
from prediction_market_agent_tooling.markets.agent_market import SortBy, FilterBy
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from prediction_market_agent_tooling.tools.utils import utcnow

from prediction_market_agent.agents.autogen_general_agent.prompts import (
    INFLUENCER_PROMPT,
)
from prediction_market_agent.utils import APIKeys


# We have an influencer and a critic. Based on https://microsoft.github.io/autogen/docs/notebooks/agentchat_nestedchat/.


class AutogenAgentType(str, Enum):
    WRITER = "writer"
    CRITIC = "critic"
    USER = "user"


def reflection_message(
    recipient: UserProxyAgent,
    messages: list[Dict],
    sender: AssistantAgent,
    config: Optional[Any] = None,
):
    print("Reflecting...", "yellow")
    return f"""
        Reflect and provide critique on the following tweet. \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}
        Note that it should not include inappropriate language.
        Note also that the tweet should not sound robotic, instead as human-like as possible.
        Also make sure to ask the recipient for an improved version of the tweet, following your critic, and nothing else.
        """


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
        # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    )

    writer = autogen.AssistantAgent(
        name="Writer",
        llm_config=llm_config,
        system_message="""
        You are a professional writer, known for your insightful and engaging articles.
        You transform complex concepts into compelling narratives.
        You should imporve the quality of the content based on the feedback from the user.
        """,
    )

    critic = autogen.AssistantAgent(
        name="Critic",
        llm_config=llm_config,
        system_message="""
            You are a critic, known for your thoroughness and commitment to standards.
            Your task is to scrutinize content for any harmful elements or regulatory violations, ensuring
            all materials align with required guidelines.
            For code
            """,
    )

    return {
        AutogenAgentType.CRITIC: critic,
        AutogenAgentType.WRITER: writer,
        AutogenAgentType.USER: user_proxy,
    }


def build_tweet(model: str) -> str:
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
        trigger=writer,  # condition=my_condition,
    )

    sh = OmenSubgraphHandler()
    one_year_ago = utcnow() - datetime.timedelta(days=365)
    markets_closing_sooner = sh.get_omen_binary_markets_simple(
        limit=5,
        sort_by=SortBy.CLOSING_SOONEST,
        filter_by=FilterBy.OPEN,
        created_after=one_year_ago,
    )
    print(f"closing markets {markets_closing_sooner}")

    task = Template(INFLUENCER_PROMPT).substitute(
        QUESTIONS=[
            {"question": m.question_title, "likelihood": m.current_p_yes}
            for m in markets_closing_sooner
        ]
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
    logger.debug(f"Cast to post - {tweet}")
    return tweet
