from prediction_market_agent_tooling.gtypes import ChecksumAddress
from pydantic import BaseModel
from web3 import Web3

from prediction_market_agent.agents.identifiers import AgentIdentifier


class NFTAgent(BaseModel):
    name: str
    identifier: AgentIdentifier
    wallet_address: ChecksumAddress
    mech_address: ChecksumAddress

    @property
    def url(self) -> str:
        return self.name.lower().replace(" ", "-")


AGENTS_IN_THE_GAME = [
    NFTAgent(
        name="Sneaky Pete",
        identifier=AgentIdentifier.MICROCHAIN_AGENT_OMEN_LEARNING_0,
        wallet_address=Web3.to_checksum_address(
            "0xd845A24014B3BD96212A21f602a4F16A7dA518A4"
        ),
        mech_address=Web3.to_checksum_address(
            "0xDDe0780F744B84b505E344931F37cEDEaD8B6163"
        ),
    ),
    NFTAgent(
        name="Banana Man",
        identifier=AgentIdentifier.NFT_GAME_AGENT_1,
        wallet_address=Web3.to_checksum_address(
            "0xb4D8C8BedE2E49b08d2A22485f72fA516116FE7F"
        ),
        mech_address=Web3.to_checksum_address(
            "0xEB98bfB88b469B60EE165F7e07c8450145999831"
        ),
    ),
    NFTAgent(
        name="Whispers",
        identifier=AgentIdentifier.NFT_GAME_AGENT_2,
        wallet_address=Web3.to_checksum_address(
            "0xC09a8aB38A554022ACBACBA174F14C8B35E89946"
        ),
        mech_address=Web3.to_checksum_address(
            "0x5CF37d5A367fcb49F49Cbb2F012b0c0748559D98"
        ),
    ),
    NFTAgent(
        name="Fuzzy Feet",
        identifier=AgentIdentifier.NFT_GAME_AGENT_3,
        wallet_address=Web3.to_checksum_address(
            "0xd4fC4305DC1226c38356024c26cdE985817f137F"
        ),
        mech_address=Web3.to_checksum_address(
            "0x34c96c1abf80787c389B8d9f2C5Cb1E7C435D43B"
        ),
    ),
    NFTAgent(
        name="Bubble Beard",
        identifier=AgentIdentifier.NFT_GAME_AGENT_4,
        wallet_address=Web3.to_checksum_address(
            "0x84690A78d74e90608fc3e73cA79A06ee4F261A06"
        ),
        mech_address=Web3.to_checksum_address(
            "0x519c0eC90Dd217A70BA024Ee7a6390b856A69Af6"
        ),
    ),
]
TREASURY_SAFE_ADDRESS = Web3.to_checksum_address(
    "0xd1A54FD7a200C2ca76B6D06437795d660d37FE28"
)
NFT_KEYS_ADDRESS = Web3.to_checksum_address(
    "0x0D7C0Bd4169D090038c6F41CFd066958fe7619D0"
)
