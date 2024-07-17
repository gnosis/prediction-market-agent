"""
Entrypoint for running the agent in GKE.
If the agent adheres to PMAT standard (subclasses deployable agent), 
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
from prediction_market_agent.agents.metaculus_agent.deploy import (
    DeployableMetaculusBotTournamentAgent,
)
from prediction_market_agent.agents.microchain_agent.deploy import (
    DeployableMicrochainAgent,
    DeployableMicrochainModifiableSystemPromptAgent0,
    DeployableMicrochainModifiableSystemPromptAgent1,
    DeployableMicrochainModifiableSystemPromptAgent2,
)
from prediction_market_agent.agents.replicate_to_omen_agent.deploy import (
    DeployableReplicateToOmenAgent,
)
from prediction_market_agent.agents.social_media_agent.deploy import (
    DeployableSocialMediaAgent,
)
from prediction_market_agent.agents.think_thoroughly_agent.deploy import (
    DeployableThinkThoroughlyAgent,
    DeployableThinkThoroughlyProphetResearchAgent,
)


class RunnableAgent(str, Enum):
    coinflip = "coinflip"
    replicate_to_omen = "replicate_to_omen"
    think_thoroughly = "think_thoroughly"
    think_thoroughly_prophet = "think_thoroughly_prophet"
    knownoutcome = "knownoutcome"
    microchain = "microchain"
    microchain_modifiable_system_prompt_0 = "microchain_modifiable_system_prompt_0"
    microchain_modifiable_system_prompt_1 = "microchain_modifiable_system_prompt_1"
    microchain_modifiable_system_prompt_2 = "microchain_modifiable_system_prompt_2"
    metaculus_bot_tournament_agent = "metaculus_bot_tournament_agent"
    # Social media (Farcaster + Twitter)
    social_media = "social_media"


RUNNABLE_AGENTS = {
    RunnableAgent.coinflip: DeployableCoinFlipAgent,
    RunnableAgent.replicate_to_omen: DeployableReplicateToOmenAgent,
    RunnableAgent.think_thoroughly: DeployableThinkThoroughlyAgent,
    RunnableAgent.think_thoroughly_prophet: DeployableThinkThoroughlyProphetResearchAgent,
    RunnableAgent.knownoutcome: DeployableKnownOutcomeAgent,
    RunnableAgent.microchain: DeployableMicrochainAgent,
    RunnableAgent.microchain_modifiable_system_prompt_0: DeployableMicrochainModifiableSystemPromptAgent0,
    RunnableAgent.microchain_modifiable_system_prompt_1: DeployableMicrochainModifiableSystemPromptAgent1,
    RunnableAgent.microchain_modifiable_system_prompt_2: DeployableMicrochainModifiableSystemPromptAgent2,
    RunnableAgent.social_media: DeployableSocialMediaAgent,
    RunnableAgent.metaculus_bot_tournament_agent: DeployableMetaculusBotTournamentAgent,
}

APP = typer.Typer(pretty_exceptions_enable=False)


@APP.command()
def main(agent: RunnableAgent, market_type: MarketType) -> None:
    RUNNABLE_AGENTS[agent]().run(market_type)


if __name__ == "__main__":
    APP()
