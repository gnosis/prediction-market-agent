"""
Entrypoint for running the agent in GKE.
If the agent adheres to PMAT standard (subclasses DeployableAgent), 
simply add the agent to the `RunnableAgent` enum and then `RUNNABLE_AGENTS` dict.
"""

import typer

from enum import Enum
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent.agents.coinflip_agent.coinflip_agent import (
    DeployableCoinFlipAgent,
)


class RunnableAgent(str, Enum):
    coinflip = "coinflip"


RUNNABLE_AGENTS = {
    RunnableAgent.coinflip: DeployableCoinFlipAgent,
}


def main(agent: RunnableAgent, market_type: MarketType):
    RUNNABLE_AGENTS[agent]().run(market_type)


if __name__ == "__main__":
    typer.run(main)
