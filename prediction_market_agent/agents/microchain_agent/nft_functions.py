from microchain import Function
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.contract import (
    ContractOwnableERC721OnGnosisChain,
)
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.microchain_agent_keys import (
    MicrochainAgentKeys,
)


class BalanceOfNFT(Function):
    @property
    def description(self) -> str:
        return "Returns the number of given NFT owned by the specified address."

    @property
    def example_args(self) -> list[str]:
        return [
            "0xNFTAddress",
            "0xOwnerddress",
        ]

    def __call__(
        self,
        nft_address: str,
        owner_address: str,
    ) -> int:
        contract = ContractOwnableERC721OnGnosisChain(
            address=Web3.to_checksum_address(nft_address)
        )
        balance: int = contract.balanceOf(Web3.to_checksum_address(owner_address))
        return balance


class OwnerOfNFT(Function):
    @property
    def description(self) -> str:
        return "Returns the owner address of the specified NFT token ID."

    @property
    def example_args(self) -> list[str]:
        return ["0xNFTAddress", "1"]

    def __call__(
        self,
        nft_address: str,
        token_id: int,
    ) -> str:
        contract = ContractOwnableERC721OnGnosisChain(
            address=Web3.to_checksum_address(nft_address)
        )
        owner_address: str = contract.owner_of(token_id)
        return owner_address


class SafeTransferFromNFT(Function):
    @property
    def description(self) -> str:
        return "Transfers the specified NFT token ID from one address to another."

    @property
    def example_args(self) -> list[str]:
        return [
            "0xNFTAddress",
            "0xRecipientAddress",
            "1",
        ]

    def __call__(
        self,
        nft_address: str,
        to_address: str,
        token_id: int,
    ) -> str:
        keys = MicrochainAgentKeys()
        contract = ContractOwnableERC721OnGnosisChain(
            address=Web3.to_checksum_address(nft_address)
        )
        if keys.ENABLE_NFT_TRANSFER:
            contract.safeTransferFrom(
                api_keys=keys,
                from_address=keys.bet_from_address,
                to_address=Web3.to_checksum_address(to_address),
                tokenId=token_id,
            )
        else:
            logger.warning("NFT transfer is disabled in the environment.")
        return "Token transferred successfully."


NFT_FUNCTIONS: list[type[Function]] = [
    BalanceOfNFT,
    OwnerOfNFT,
    SafeTransferFromNFT,
]
