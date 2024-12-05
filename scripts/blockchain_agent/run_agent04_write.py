import asyncio

from autogen_agentchat.task import Console

from prediction_market_agent.agents.blockchain_coding_agent.agents import get_agent_team
from prediction_market_agent.agents.blockchain_coding_agent.prompts import (
    SEND_ONCHAIN_FUNCTION_PROMPT,
)


async def main() -> None:
    agent_team = get_agent_team()
    stream = agent_team.run_stream(task=SEND_ONCHAIN_FUNCTION_PROMPT)
    await Console(stream)


asyncio.run(main())
