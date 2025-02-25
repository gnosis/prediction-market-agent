"""
Entrypoint for running the agent in GKE.
If the agent adheres to PMAT standard (subclasses deployable agent), 
simply add the agent to the `RunnableAgent` enum and then `RUNNABLE_AGENTS` dict.

Can also be executed locally, simply by running `python prediction_market_agent/run_agent.py <agent> <market_type>`.
"""

from enum import Enum

import nest_asyncio
import typer
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.arbitrage_agent.deploy import (
    DeployableArbitrageAgent,
)
from prediction_market_agent.agents.coinflip_agent.deploy import DeployableCoinFlipAgent
from prediction_market_agent.agents.invalid_agent.deploy import InvalidAgent
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
    DeployableMicrochainModifiableSystemPromptAgent3,
    DeployableMicrochainWithGoalManagerAgent0,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.deploy_nft_treasury_game import (
    DeployableAgentNFTGame1,
    DeployableAgentNFTGame2,
    DeployableAgentNFTGame3,
    DeployableAgentNFTGame4,
    DeployableAgentNFTGame5,
    DeployableAgentNFTGame6,
    DeployableAgentNFTGame7,
)
from prediction_market_agent.agents.ofvchallenger_agent.deploy import OFVChallengerAgent
from prediction_market_agent.agents.omen_cleaner_agent.deploy import OmenCleanerAgent
from prediction_market_agent.agents.prophet_agent.deploy import (
    DeployableOlasEmbeddingOAAgent,
    DeployablePredictionProphetClaude3OpusAgent,
    DeployablePredictionProphetClaude35HaikuAgent,
    DeployablePredictionProphetClaude35SonnetAgent,
    DeployablePredictionProphetGPT4oAgent,
    DeployablePredictionProphetGPT4oAgent_B,
    DeployablePredictionProphetGemini20Flash,
    DeployablePredictionProphetDeepSeekR1,
    DeployablePredictionProphetDeepSeekChat,
    DeployablePredictionProphetGPT4oAgentNewMarketTrader,
    DeployablePredictionProphetGPT4ominiAgent,
    DeployablePredictionProphetGPT4TurboFinalAgent,
    DeployablePredictionProphetGPT4TurboPreviewAgent,
    DeployablePredictionProphetGPTo1,
    DeployablePredictionProphetGPTo1MiniAgent,
    DeployablePredictionProphetGPTo1PreviewAgent,
    DeployablePredictionProphetGPTo3mini,
)
from prediction_market_agent.agents.replicate_to_omen_agent.deploy import (
    DeployableReplicateToOmenAgent,
)
from prediction_market_agent.agents.social_media_agent.deploy import (
    DeployableSocialMediaAgent,
)
from prediction_market_agent.agents.specialized_agent.deploy import (
    MarketCreatorsStalkerAgent1,
    MarketCreatorsStalkerAgent2,
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
    microchain_modifiable_system_prompt_3 = "microchain_modifiable_system_prompt_3"
    microchain_with_goal_manager_agent_0 = "microchain_with_goal_manager_agent_0"
    metaculus_bot_tournament_agent = "metaculus_bot_tournament_agent"
    prophet_gpt4o = "prophet_gpt4o"
    prophet_gpt4o_b = "prophet_gpt4o_b"
    prophet_gpt4o_new_market_trader = "prophet_gpt4o_new_market_trader"
    prophet_gpt4 = "prophet_gpt4"
    prophet_gpt4_final = "prophet_gpt4_final"
    prophet_o1preview = "prophet_o1preview"
    prophet_o1mini = "prophet_o1mini"
    prophet_o1 = "prophet_o1"
    prophet_o3mini = "prophet_o3mini"
    prophet_gpt4omini = "prophet_gpt4omini"
    prophet_gemini20flash = "prophet_gemini20flash"
    prophet_deekseekchat = "prophet_deekseekchat"
    prophet_deepseekr1 = "prophet_deepseekr1"
    olas_embedding_oa = "olas_embedding_oa"
    # Social media (Farcaster + Twitter)
    social_media = "social_media"
    omen_cleaner = "omen_cleaner"
    ofv_challenger = "ofv_challenger"
    arbitrage = "arbitrage"
    market_creators_stalker1 = "market_creators_stalker1"
    market_creators_stalker2 = "market_creators_stalker2"
    invalid = "invalid"
    nft_treasury_game_agent_1 = "nft_treasury_game_agent_1"
    nft_treasury_game_agent_2 = "nft_treasury_game_agent_2"
    nft_treasury_game_agent_3 = "nft_treasury_game_agent_3"
    nft_treasury_game_agent_4 = "nft_treasury_game_agent_4"
    nft_treasury_game_agent_5 = "nft_treasury_game_agent_5"
    nft_treasury_game_agent_6 = "nft_treasury_game_agent_6"
    nft_treasury_game_agent_7 = "nft_treasury_game_agent_7"
    prophet_claude3_opus = "prophet_claude3_opus"
    prophet_claude35_haiku = "prophet_claude35_haiku"
    prophet_claude35_sonnet = "prophet_claude35_sonnet"


RUNNABLE_AGENTS: dict[RunnableAgent, type[DeployableAgent]] = {
    RunnableAgent.coinflip: DeployableCoinFlipAgent,
    RunnableAgent.replicate_to_omen: DeployableReplicateToOmenAgent,
    RunnableAgent.think_thoroughly: DeployableThinkThoroughlyAgent,
    RunnableAgent.think_thoroughly_prophet: DeployableThinkThoroughlyProphetResearchAgent,
    RunnableAgent.knownoutcome: DeployableKnownOutcomeAgent,
    RunnableAgent.microchain: DeployableMicrochainAgent,
    RunnableAgent.microchain_modifiable_system_prompt_0: DeployableMicrochainModifiableSystemPromptAgent0,
    RunnableAgent.microchain_modifiable_system_prompt_1: DeployableMicrochainModifiableSystemPromptAgent1,
    RunnableAgent.microchain_modifiable_system_prompt_2: DeployableMicrochainModifiableSystemPromptAgent2,
    RunnableAgent.microchain_modifiable_system_prompt_3: DeployableMicrochainModifiableSystemPromptAgent3,
    RunnableAgent.microchain_with_goal_manager_agent_0: DeployableMicrochainWithGoalManagerAgent0,
    RunnableAgent.social_media: DeployableSocialMediaAgent,
    RunnableAgent.metaculus_bot_tournament_agent: DeployableMetaculusBotTournamentAgent,
    RunnableAgent.prophet_gpt4o: DeployablePredictionProphetGPT4oAgent,
    RunnableAgent.prophet_gpt4o_new_market_trader: DeployablePredictionProphetGPT4oAgentNewMarketTrader,
    RunnableAgent.prophet_gpt4: DeployablePredictionProphetGPT4TurboPreviewAgent,
    RunnableAgent.prophet_gpt4_final: DeployablePredictionProphetGPT4TurboFinalAgent,
    RunnableAgent.olas_embedding_oa: DeployableOlasEmbeddingOAAgent,
    RunnableAgent.omen_cleaner: OmenCleanerAgent,
    RunnableAgent.ofv_challenger: OFVChallengerAgent,
    RunnableAgent.prophet_o1preview: DeployablePredictionProphetGPTo1PreviewAgent,
    RunnableAgent.prophet_o1mini: DeployablePredictionProphetGPTo1MiniAgent,
    RunnableAgent.prophet_o1: DeployablePredictionProphetGPTo1,
    RunnableAgent.prophet_o3mini: DeployablePredictionProphetGPTo3mini,
    RunnableAgent.prophet_gpt4omini: DeployablePredictionProphetGPT4ominiAgent,
    RunnableAgent.arbitrage: DeployableArbitrageAgent,
    RunnableAgent.market_creators_stalker1: MarketCreatorsStalkerAgent1,
    RunnableAgent.market_creators_stalker2: MarketCreatorsStalkerAgent2,
    RunnableAgent.invalid: InvalidAgent,
    RunnableAgent.nft_treasury_game_agent_1: DeployableAgentNFTGame1,
    RunnableAgent.nft_treasury_game_agent_2: DeployableAgentNFTGame2,
    RunnableAgent.nft_treasury_game_agent_3: DeployableAgentNFTGame3,
    RunnableAgent.nft_treasury_game_agent_4: DeployableAgentNFTGame4,
    RunnableAgent.nft_treasury_game_agent_5: DeployableAgentNFTGame5,
    RunnableAgent.nft_treasury_game_agent_6: DeployableAgentNFTGame6,
    RunnableAgent.nft_treasury_game_agent_7: DeployableAgentNFTGame7,
    RunnableAgent.prophet_claude3_opus: DeployablePredictionProphetClaude3OpusAgent,
    RunnableAgent.prophet_claude35_haiku: DeployablePredictionProphetClaude35HaikuAgent,
    RunnableAgent.prophet_claude35_sonnet: DeployablePredictionProphetClaude35SonnetAgent,
    RunnableAgent.prophet_gpt4o_b: DeployablePredictionProphetGPT4oAgent_B,
    RunnableAgent.prophet_gemini20flash: DeployablePredictionProphetGemini20Flash,
    RunnableAgent.prophet_deepseekr1: DeployablePredictionProphetDeepSeekR1,
    RunnableAgent.prophet_deekseekchat: DeployablePredictionProphetDeepSeekChat,
}

APP = typer.Typer(pretty_exceptions_enable=False)


@APP.command()
def main(
    agent: RunnableAgent,
    market_type: MarketType,
) -> None:
    nest_asyncio.apply()  # See https://github.com/pydantic/pydantic-ai/issues/889, we had issue with Think Thoroughly that is using multiprocessing heavily.
    RUNNABLE_AGENTS[agent]().run(market_type=market_type)


if __name__ == "__main__":
    APP()
