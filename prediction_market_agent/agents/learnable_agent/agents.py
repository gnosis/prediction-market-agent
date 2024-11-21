# The code writer agent's system message is to instruct the LLM on how to use
# the code executor in the code executor agent.
from pathlib import Path

import streamlit as st
from autogen import ConversableAgent
from autogen.coding.jupyter import LocalJupyterServer, JupyterCodeExecutor

from prediction_market_agent.agents.learnable_agent.prompts import (
    code_writer_system_message,
)
from prediction_market_agent.utils import APIKeys

MAX_CONSECUTIVE_AUTO_REPLY = 30


def get_code_executor_agent(for_streamlit: bool = False) -> ConversableAgent:
    output_dir = Path("coding")
    output_dir.mkdir(exist_ok=True)
    params = dict(
        name="code_executor_agent",
        llm_config=False,  # Turn off LLM for this agent.
        code_execution_config={
            "executor": JupyterCodeExecutor(LocalJupyterServer(), output_dir=output_dir)
        },  # Use the local command line code executor.
        human_input_mode="NEVER",
        max_consecutive_auto_reply=MAX_CONSECUTIVE_AUTO_REPLY,
    )
    if for_streamlit:
        code_executor_agent = TrackableConversableAgent(**params)
    else:
        code_executor_agent = ConversableAgent(**params)

    return code_executor_agent


def get_code_writer_agent(for_streamlit: bool = False) -> ConversableAgent:
    llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": APIKeys().openai_api_key.get_secret_value(),
            }
        ]
    }
    params = dict(
        name="code_writer_agent",
        system_message=code_writer_system_message,
        llm_config=llm_config,
        code_execution_config=False,  # Turn off code execution for this agent.
    )
    if for_streamlit:
        code_writer_agent = TrackableConversableAgent(**params)
    else:
        code_writer_agent = ConversableAgent(**params)

    return code_writer_agent


class TrackableConversableAgent(ConversableAgent):
    """Helper class for displaying intermediate messages in Streamlit."""

    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)


class TrackableAssistantAgent(TrackableConversableAgent):
    pass


class TrackableUserProxyAgent(TrackableConversableAgent):
    pass
