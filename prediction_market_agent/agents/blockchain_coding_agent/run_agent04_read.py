# Define a tool
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.task import Console, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models import OpenAIChatCompletionClient

from prediction_market_agent.agents.blockchain_coding_agent.functions import (
    fetch_source_code_and_abi_from_contract,
)
from prediction_market_agent.utils import APIKeys


async def main() -> None:
    # Define an agent
    weather_agent = AssistantAgent(
        name="blockchain_expert_agent",
        model_client=OpenAIChatCompletionClient(
            model="gpt-4o",
            api_key=APIKeys().openai_api_key.get_secret_value(),
        ),
        tools=[
            a_get_rpc_endpoint,
            a_checksum_address,
            a_execute_read_function,
            fetch_source_code_and_abi_from_contract,
        ],
    )

    # Define termination condition
    termination = TextMentionTermination("TERMINATE")

    # Define a team
    agent_team = RoundRobinGroupChat([weather_agent], termination_condition=termination)

    # Run the team and stream messages to the console
    message_read = "Use the web3.py Python library and interact with the Conditional Tokens contract on the Gnosis Chain (contract address 0xCeAfDD6bc0bEF976fdCd1112955828E00543c0Ce) in order to read the balance of wallet address 0x2FC96c4e7818dBdc3D72A463F47f0E1CeEa0A2D0 with position id 38804060408381130621475891941405037249059836800475827360004002125093421139610. Return the balance fetched using the latest block. Consider using the function execute_read_function to execute a read function on the smart contract. Whenever passing an address as parameter, calculate the checksum address of the address. Let's think step-by-step."
    stream = agent_team.run_stream(task=message_read)
    await Console(stream)


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run(main())
