import datetime
from typing import Dict, Optional, Any

import typer
from autogen import AssistantAgent, UserProxyAgent
from autogen.cache import Cache
from prediction_market_agent_tooling.markets.agent_market import SortBy, FilterBy
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from prediction_market_agent_tooling.tools.utils import utcnow

from prediction_market_agent.agents.autogen_general_agent.agents import (
    user_proxy,
    critic,
    writer,
)


# We have an influencer and a critic. Based on https://microsoft.github.io/autogen/docs/notebooks/agentchat_nestedchat/.


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
    Also make sure to ask the recipient for an improved version of the tweet, following your critic, and nothing else.
    """


def main():
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

    task = f"""
    You are an influencer that likes to share your view of the world, based on recent events.
    You can use the following QUESTIONS about recent events as inspiration for your tweet.
    
    [QUESTIONS]
    {[{"question": m.question_title, "likelihood": m.current_p_yes} for m in markets_closing_sooner]}
    
    Write an engaging tweet about recent topics that you think your audience will be interested in.
    Do not add any reasoning or additional explanation, simply output the tweet.
    """
    with Cache.disk(cache_seed=42) as cache:
        # max_turns = the maximum number of turns for the chat between the two agents. One turn means one conversation round trip.
        res = user_proxy.initiate_chat(
            recipient=writer,
            message=task,
            max_turns=2,
            summary_method="last_msg",
            cache=cache,
        )

    print(f"tweet {res.summary}")


if __name__ == "__main__":
    typer.run(main)
