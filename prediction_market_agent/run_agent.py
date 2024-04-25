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
from prediction_market_agent.agents.known_outcome_agent.deploy import (
    DeployableKnownOutcomeAgent,
)
from prediction_market_agent.agents.mech_agent.deploy import (
    DeployablePredictionOfflineAgent,
    DeployablePredictionOnlineAgent,
)
from prediction_market_agent.agents.replicate_to_omen_agent.deploy import (
    DeployableReplicateToOmenAgent,
)
from prediction_market_agent.agents.think_thoroughly_agent.deploy import (
    DeployableThinkThoroughlyAgent,
)


class RunnableAgent(str, Enum):
    coinflip = "coinflip"
    replicate_to_omen = "replicate_to_omen"
    think_thoroughly = "think_thoroughly"
    knownoutcome = "knownoutcome"
    # Mechs
    mech_prediction_online = "mech_prediction-online"
    mech_prediction_offline = "mech_prediction-offline"


RUNNABLE_AGENTS = {
    RunnableAgent.coinflip: DeployableCoinFlipAgent,
    RunnableAgent.replicate_to_omen: DeployableReplicateToOmenAgent,
    RunnableAgent.think_thoroughly: DeployableThinkThoroughlyAgent,
    RunnableAgent.knownoutcome: DeployableKnownOutcomeAgent,
    RunnableAgent.mech_prediction_online: DeployablePredictionOnlineAgent,
    RunnableAgent.mech_prediction_offline: DeployablePredictionOfflineAgent,
}


def main(agent: RunnableAgent, market_type: MarketType) -> None:
    RUNNABLE_AGENTS[agent]().run(market_type)


if __name__ == "__main__":
    typer.run(main)
