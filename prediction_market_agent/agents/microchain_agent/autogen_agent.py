import os

import typer
from autogen import register_function, ConversableAgent

from prediction_market_agent.agents.microchain_agent.autogen_functions import (
    sum_function,
)
from prediction_market_agent.utils import APIKeys


def main() -> None:
    keys = APIKeys()
    assistant = ConversableAgent(
        name="Assistant",
        system_message="You are a helpful AI assistant. "
        "You can help with simple calculations. "
        "Return 'TERMINATE' when the task is done.",
        llm_config={
            "config_list": [
                {"model": "gpt-4", "api_key": keys.openai_api_key.get_secret_value()}
            ]
        },
    )

    user_proxy = ConversableAgent(
        name="User",
        llm_config=False,
        is_termination_msg=lambda msg: msg.get("content") is not None
        and "TERMINATE" in msg["content"],
        human_input_mode="NEVER",
    )

    register_function(
        sum_function,
        caller=assistant,  # The assistant agent can suggest calls to the calculator.
        executor=user_proxy,  # The user proxy agent can execute the calculator calls.
        name="sum_function",  # By default, the function name is used as the tool name.
        description="Adds two numbers together",  # A description of the tool.
    )

    chat_result = user_proxy.initiate_chat(
        assistant,
        message="What is 44232 + 13312? Answer only with the numerical output.",
    )
    print(f"sum = {chat_result.summary}")


if __name__ == "__main__":
    typer.run(main)
