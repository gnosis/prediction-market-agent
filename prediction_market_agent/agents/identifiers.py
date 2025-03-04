from typing import NewType

AgentIdentifier = NewType("AgentIdentifier", str)

THINK_THOROUGHLY = AgentIdentifier("think-thoroughly-agent")
THINK_THOROUGHLY_PROPHET = AgentIdentifier("think-thoroughly-prophet-research-agent")
MICROCHAIN_AGENT_OMEN = AgentIdentifier("microchain-agent-deployment-omen")
MICROCHAIN_AGENT_OMEN_TEST = AgentIdentifier("microchain-agent-deployment-omen_test")
MICROCHAIN_AGENT_OMEN_LEARNING_0 = AgentIdentifier("general-agent-0")
MICROCHAIN_AGENT_OMEN_LEARNING_1 = AgentIdentifier("general-agent-1")
MICROCHAIN_AGENT_OMEN_LEARNING_2 = AgentIdentifier("general-agent-2")
MICROCHAIN_AGENT_OMEN_LEARNING_3 = AgentIdentifier("general-agent-3")
MICROCHAIN_AGENT_STREAMLIT = AgentIdentifier("microchain-streamlit-app")
MICROCHAIN_AGENT_OMEN_WITH_GOAL_MANAGER = AgentIdentifier(
    "trader-agent-0-with-goal-manager"
)
NFT_TREASURY_GAME_AGENT_1 = AgentIdentifier("nft-treasury-game-agent-1")
NFT_TREASURY_GAME_AGENT_2 = AgentIdentifier("nft-treasury-game-agent-2")
NFT_TREASURY_GAME_AGENT_3 = AgentIdentifier("nft-treasury-game-agent-3")
NFT_TREASURY_GAME_AGENT_4 = AgentIdentifier("nft-treasury-game-agent-4")
NFT_TREASURY_GAME_AGENT_5 = AgentIdentifier("nft-treasury-game-agent-5")
NFT_TREASURY_GAME_AGENT_6 = AgentIdentifier("nft-treasury-game-agent-6")
NFT_TREASURY_GAME_AGENT_7 = AgentIdentifier("nft-treasury-game-agent-7")
