"""
Entrypoint for running the agent in GKE.
If the agent adheres to PMAT standard (subclasses DeployableAgent), 
simply add the agent to the `RunnableAgent` enum and then `RUNNABLE_AGENTS` dict.

Can also be executed locally, simply by running `python prediction_market_agent/run_agent.py <agent> <market_type>`.
"""

from enum import Enum

import typer
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.coinflip_agent.deploy import DeployableCoinFlipAgent


class RunnableAgent(str, Enum):
    coinflip = "coinflip"


RUNNABLE_AGENTS = {
    RunnableAgent.coinflip: DeployableCoinFlipAgent,
}


def main(agent: RunnableAgent, market_type: MarketType) -> None:
    RUNNABLE_AGENTS[agent]().run(market_type)


if __name__ == "__main__":
    typer.run(main)
