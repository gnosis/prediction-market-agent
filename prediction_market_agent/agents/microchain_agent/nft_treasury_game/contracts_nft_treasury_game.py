from functools import cache

from prediction_market_agent_tooling.gtypes import ChecksumAddress
from prediction_market_agent_tooling.tools.contract import (
    ContractOwnableERC721OnGnosisChain,
)
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    NFT_TOKEN_FACTORY,
)


class ContractNFTFactoryOnGnosisChain(ContractOwnableERC721OnGnosisChain):
    address: ChecksumAddress = NFT_TOKEN_FACTORY

    def max_supply(self, web3: Web3 | None = None) -> int:
        n_tokens: int = self.call("MAX_SUPPLY", web3=web3)
        return n_tokens

    def token_ids_owned_by(
        self, owner: ChecksumAddress, web3: Web3 | None = None
    ) -> list[int]:
        token_ids = list(range(self.max_supply(web3=web3)))
        return [
            token_id
            for token_id in token_ids
            if self.owner_of(token_id=token_id, web3=web3) == owner
        ]


@cache
def get_nft_token_factory_max_supply() -> int:
    return ContractNFTFactoryOnGnosisChain().max_supply()
