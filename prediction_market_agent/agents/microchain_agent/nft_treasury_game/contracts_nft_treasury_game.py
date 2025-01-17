from prediction_market_agent_tooling.gtypes import ChecksumAddress
from prediction_market_agent_tooling.tools.contract import (
    ContractOwnableERC721OnGnosisChain,
)

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    NFT_TOKEN_FACTORY,
)


class ContractNFTFactoryOnGnosisChain(ContractOwnableERC721OnGnosisChain):
    address: ChecksumAddress = NFT_TOKEN_FACTORY
