from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.task import TextMentionTermination
from autogen_agentchat.teams import BaseGroupChat, RoundRobinGroupChat
from autogen_ext.models._openai._openai_client import OpenAIChatCompletionClient

from prediction_market_agent.agents.blockchain_coding_agent.functions import (
    checksum_address,
    execute_read_function,
    execute_write_function,
    fetch_source_code_and_abi_from_contract,
    get_rpc_endpoint,
)
from prediction_market_agent.utils import APIKeys

MAX_CONSECUTIVE_AUTO_REPLY = 30


def get_blockchain_agent() -> BaseChatAgent:
    blockchain_agent = AssistantAgent(
        name="blockchain_expert_agent",
        model_client=OpenAIChatCompletionClient(
            model="gpt-4o",
            api_key=APIKeys().openai_api_key.get_secret_value(),
        ),
        tools=[
            get_rpc_endpoint,
            checksum_address,
            execute_read_function,
            execute_write_function,
            fetch_source_code_and_abi_from_contract,
        ],
    )
    return blockchain_agent


def get_agent_team() -> BaseGroupChat:
    termination = TextMentionTermination("TERMINATE")

    blockchain_agent = get_blockchain_agent()
    # Define a team
    agent_team = RoundRobinGroupChat(
        [blockchain_agent], termination_condition=termination
    )
    return agent_team
