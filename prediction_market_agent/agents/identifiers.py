from enum import Enum


class AgentIdentifier(str, Enum):
    THINK_THOROUGHLY = "think-thoroughly-agent"
    THINK_THOROUGHLY_PROPHET = "think-thoroughly-prophet-research-agent"
    MICROCHAIN_AGENT_OMEN = "microchain-agent-deployment-omen"
    MICROCHAIN_AGENT_OMEN_TEST = "microchain-agent-deployment-omen_test"
    MICROCHAIN_AGENT_OMEN_LEARNING_0 = "general-agent-0"
    MICROCHAIN_AGENT_OMEN_LEARNING_1 = "general-agent-1"
    MICROCHAIN_AGENT_OMEN_LEARNING_2 = "general-agent-2"
    MICROCHAIN_AGENT_OMEN_LEARNING_3 = "general-agent-3"
    MICROCHAIN_AGENT_STREAMLIT = "microchain-streamlit-app"
    MICROCHAIN_AGENT_OMEN_WITH_GOAL_MANAGER = "trader-agent-0-with-goal-manager"
    NFT_TREASURY_GAME_AGENT_1 = "nft-treasury-game-agent-1"
    NFT_TREASURY_GAME_AGENT_2 = "nft-treasury-game-agent-2"
    NFT_TREASURY_GAME_AGENT_3 = "nft-treasury-game-agent-3"
    NFT_TREASURY_GAME_AGENT_4 = "nft-treasury-game-agent-4"
    NFT_TREASURY_GAME_AGENT_5 = "nft-treasury-game-agent-5"
    NFT_TREASURY_GAME_AGENT_6 = "nft-treasury-game-agent-6"
    NFT_TREASURY_GAME_AGENT_7 = "nft-treasury-game-agent-7"
