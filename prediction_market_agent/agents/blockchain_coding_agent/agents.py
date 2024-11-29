# The code writer agent's system message is to instruct the LLM on how to use
# the code executor in the code executor agent.
import typing as t
from pathlib import Path

import chromadb
import streamlit as st
from autogen import Agent, ConversableAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.vectordb.chromadb import ChromaVectorDB
from autogen.coding import LocalCommandLineCodeExecutor
from autogen.retrieve_utils import parse_html_to_markdown
from chromadb.utils import embedding_functions

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
    model = "text-embedding-3-large"  # same model as used by PineconeHandler
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=APIKeys().openai_api_key.get_secret_value(),
        model_name=model,
    )
    vector_db = ChromaVectorDB(embedding_function=openai_ef)

    return RetrieveUserProxyAgent(
        name="Web3 Python expert",
        is_termination_msg=termination_msg,
        human_input_mode="NEVER",
        # get_or_create=True,
        retrieve_config={
            "task": "code",
            "docs_path": [
                "prediction_market_agent/agents/blockchain_coding_agent/web3py_index.html"
            ],
            "vector_db": None,
            # "client": chromadb.PersistentClient().get_or_create_collection(
            #     name="autogen_agent", embedding_function=openai_ef
            # ),
            # "chunk_mode": "one_line",  # chunk_mode
            "custom_text_split_function": parse_html_to_markdown,
            "client": chromadb.PersistentClient(),
            "embedding_function": openai_ef,
            "model": "gpt-4o",
            # "collection_name": "groupchat",
            "get_or_create": True,
        },
        code_execution_config={
            "executor": LocalCommandLineCodeExecutor(),
        },
        description="Assistant who is an expert in web3.py package documentation and has extra content retrieval power for solving coding tasks related to web3.py usage.",
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
