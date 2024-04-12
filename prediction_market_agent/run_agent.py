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
from prediction_market_agent.agents.prophet_agent.deploy import (
    DeployableOlasEmbeddingOAAgent,
    DeployablePredictionProphetGPT3Agent,
    DeployablePredictionProphetGPT4Agent,
)
from prediction_market_agent.agents.replicator_agent.deploy import (
    DeployableReplicateToOmenAgent,
)


class RunnableAgent(str, Enum):
    coinflip = "coinflip"
    prophet_gpt3 = "prophet_gpt3"
    prophet_gpt4 = "prophet_gpt4"
    olas_embedding_oa = "olas_embedding_oa"
    replicator = "replicator"


RUNNABLE_AGENTS = {
    RunnableAgent.coinflip: DeployableCoinFlipAgent,
    RunnableAgent.prophet_gpt3: DeployablePredictionProphetGPT3Agent,
    RunnableAgent.prophet_gpt4: DeployablePredictionProphetGPT4Agent,
    RunnableAgent.olas_embedding_oa: DeployableOlasEmbeddingOAAgent,
    RunnableAgent.replicator: DeployableReplicateToOmenAgent,
}


def main(agent: RunnableAgent, market_type: MarketType) -> None:
    RUNNABLE_AGENTS[agent]().run(market_type)


if __name__ == "__main__":
    typer.run(main)
