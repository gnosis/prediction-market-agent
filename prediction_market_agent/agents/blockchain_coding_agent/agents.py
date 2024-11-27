# The code writer agent's system message is to instruct the LLM on how to use
# the code executor in the code executor agent.
import typing as t
from pathlib import Path

import streamlit as st
from autogen import Agent, ConversableAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor

from prediction_market_agent.agents.blockchain_coding_agent.prompts import (
    code_writer_system_message,
)
from prediction_market_agent.utils import APIKeys

MAX_CONSECUTIVE_AUTO_REPLY = 30


def get_code_executor_agent(for_streamlit: bool = False) -> ConversableAgent:
    local_executor = LocalCommandLineCodeExecutor()
    output_dir = Path(".cache/coding")
    output_dir.mkdir(exist_ok=True)
    code_executor_agent = (
        TrackableConversableAgent if for_streamlit else ConversableAgent
    )(
        name="code_executor_agent",
        llm_config=False,  # Turn off LLM for this agent.
        code_execution_config={
            "executor": local_executor,
        },  # Use the local command line code executor.
        human_input_mode="NEVER",
        max_consecutive_auto_reply=MAX_CONSECUTIVE_AUTO_REPLY,
    )

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

    code_writer_agent = (
        TrackableConversableAgent if for_streamlit else ConversableAgent
    )(
        name="code_writer_agent",
        system_message=code_writer_system_message,
        llm_config=llm_config,
        code_execution_config=False,
    )

    return code_writer_agent


def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


def get_code_rag_agent(for_streamlit: bool = False) -> RetrieveUserProxyAgent:
    # ToDo - Pass list of web3.py docs to the RetrieveUserProxyAgent
    return RetrieveUserProxyAgent(
        name="Boss_Assistant",
        is_termination_msg=termination_msg,
        human_input_mode="NEVER",
        default_auto_reply="Reply `TERMINATE` if the task is done.",
        max_consecutive_auto_reply=3,
        get_or_create=True,
        retrieve_config={
            "task": "code",
            "docs_path": "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
            "chunk_token_size": 1000,
            "model": "gpt-4o",
            "collection_name": "groupchat",
            "get_or_create": True,
        },
        code_execution_config=False,  # we don't want to execute code in this case.
        description="Assistant who has extra content retrieval power for solving difficult problems.",
    )


class TrackableConversableAgent(ConversableAgent):
    """Helper class for displaying intermediate messages in Streamlit."""

    def _process_received_message(
        self, message: t.Union[t.Dict[t.Any, t.Any], str], sender: Agent, silent: bool
    ) -> t.Any:
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)


class TrackableAssistantAgent(TrackableConversableAgent):
    pass


class TrackableUserProxyAgent(TrackableConversableAgent):
    pass
