import typing as t

from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.tools.contract import SimpleTreasuryContract
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from pydantic import BaseModel, Field

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.deploy_nft_treasury_game import (
    DEPLOYED_NFT_AGENTS,
)


class EventBase(BaseModel):
    block_number: int = Field(alias="blockNumber")
    transaction_hash: HexBytes = Field(alias="transactionHash")
    address: HexBytes
    event: str


class ERC721Transfer(EventBase):
    from_address: HexBytes
    to_address: HexBytes
    token_id: int

    @classmethod
    def from_event_log(cls, log: dict[t.Any, t.Any]) -> "ERC721Transfer":
        d = {
            "from_address": log["args"]["from"],
            "to_address": log["args"]["to"],
            "token_id": log["args"]["tokenId"],
        }
        return ERC721Transfer.model_validate({**d, **log})


class TransactionDict(BaseModel):
    from_address: ChecksumAddress = Field(alias="from")
    to_address: ChecksumAddress = Field(alias="to")
    block_number: int = Field(alias="blockNumber")
    hash: HexBytes
    value: int
    input: HexBytes | None
    type: int

    def relevant_to_nft_game(self) -> bool:
        agents_addresses = [a.wallet_address for a in DEPLOYED_NFT_AGENTS]
        involves_nft_agents = (
            self.from_address in agents_addresses or self.to_address in agents_addresses
        )
        treasury_address = SimpleTreasuryContract().address
        involves_treasury = (
            self.from_address == treasury_address or self.to_address == treasury_address
        )
        return involves_treasury or involves_nft_agents
