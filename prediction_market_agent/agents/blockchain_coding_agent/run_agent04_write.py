# Define a tool
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.task import Console, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models import OpenAIChatCompletionClient

from prediction_market_agent.agents.blockchain_coding_agent.functions import *
from prediction_market_agent.utils import APIKeys


async def main() -> None:
    # Define an agent
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

    # Define termination condition
    termination = TextMentionTermination("TERMINATE")

    # Define a team
    agent_team = RoundRobinGroupChat(
        [blockchain_agent], termination_condition=termination
    )

    message_write = "Use the web3.py Python library and the registered functions and interact with the USDC token contract on the Gnosis Chain (contract address 0xddafbb505ad214d7b80b1f830fccc89b60fb7a83). Approve Johnny (wallet address 0x70997970C51812dc3A010C7d01b50e0d17dc79C8) as spender of your USDC, with allowance 100 USDC. Let's think step-by-step."
    stream = agent_team.run_stream(task=message_write)
    await Console(stream)


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())
