import asyncio

from autogen_agentchat.task import Console

from prediction_market_agent.agents.blockchain_coding_agent.agents import get_agent_team


async def main() -> None:
    agent_team = get_agent_team()
    message_write = "Use the web3.py Python library and the registered functions and interact with the USDC token contract on the Gnosis Chain (contract address 0xddafbb505ad214d7b80b1f830fccc89b60fb7a83). Approve Johnny (wallet address 0x70997970C51812dc3A010C7d01b50e0d17dc79C8) as spender of your USDC, with allowance 100 USDC. Let's think step-by-step."
    stream = agent_team.run_stream(task=message_write)
    await Console(stream)


asyncio.run(main())
