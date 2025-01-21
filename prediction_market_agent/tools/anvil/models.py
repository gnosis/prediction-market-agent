import typing as t

from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.tools.contract import SimpleTreasuryContract
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from pydantic import BaseModel, Field, computed_field

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.deploy_nft_treasury_game import (
    DEPLOYED_NFT_AGENTS,
)
from prediction_market_agent.tools.message_utils import unzip_message_else_do_nothing


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


class AgentCommunicationMessage(EventBase):
    sender: HexBytes
    agent_address: HexBytes = Field(alias="agentAddress")
    message: bytes
    value: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def decoded_message(self) -> str:
        try:
            return self.message.decode("utf-8")
        except:
            return unzip_message_else_do_nothing(HexBytes(self.message).hex())

    @classmethod
    def from_event_log(cls, log: dict[t.Any, t.Any]) -> "AgentCommunicationMessage":
        d = {
            "sender": log["args"]["sender"],
            "message": log["args"]["message"],
            "value": log["args"]["value"],
            "agentAddress": log["args"]["agentAddress"],
        }
        return AgentCommunicationMessage.model_validate({**d, **log})


class BalanceData(BaseModel):
    block: int
    address: str
    balance_wei: int


# class BlockDump(BaseModel):
#     transactions: list[TransactionDump]
#
#
# class AccountDump(BaseModel):
#     pass
#
#
# class AnvilDump(BaseModel):
#     blocks: list[BalanceData]
#     accounts: list[AccountDump]


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
