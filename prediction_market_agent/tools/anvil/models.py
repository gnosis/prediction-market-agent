import typing as t

from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from pydantic import BaseModel, Field


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


# Similar to TypedTransaction (https://eth-account.readthedocs.io/en/stable/eth_account.typed_transactions.html#typed-transactions)
# but as a BaseModel.
class TransactionDict(BaseModel):
    from_address: ChecksumAddress = Field(alias="from")
    to_address: ChecksumAddress = Field(alias="to")
    block_number: int = Field(alias="blockNumber")
    hash: HexBytes
    value: int
    input: HexBytes | None
    type: int
