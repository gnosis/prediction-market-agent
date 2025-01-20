from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from pydantic import BaseModel, Field, computed_field

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
    def from_event_log(cls, log: dict) -> "ERC721Transfer":
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

    @computed_field
    @property
    def decoded_message(self) -> str:
        try:
            return self.message.decode("utf-8")
        except:
            return unzip_message_else_do_nothing(HexBytes(self.message).hex())

    @classmethod
    def from_event_log(cls, log: dict) -> "AgentCommunicationMessage":
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
