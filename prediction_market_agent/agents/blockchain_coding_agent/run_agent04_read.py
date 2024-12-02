import asyncio

from autogen_agentchat.task import Console

from prediction_market_agent.agents.blockchain_coding_agent.agents import get_agent_team


async def main() -> None:
    agent_team = get_agent_team()
    message_read = "Use the web3.py Python library and interact with the Conditional Tokens contract on the Gnosis Chain (contract address 0xCeAfDD6bc0bEF976fdCd1112955828E00543c0Ce) in order to read the balance of wallet address 0x2FC96c4e7818dBdc3D72A463F47f0E1CeEa0A2D0 with position id 38804060408381130621475891941405037249059836800475827360004002125093421139610. Return the balance fetched using the latest block. Consider using the function execute_read_function to execute a read function on the smart contract. Whenever passing an address as parameter, calculate the checksum address of the address. Let's think step-by-step."
    stream = agent_team.run_stream(task=message_read)
    await Console(stream)


asyncio.run(main())
