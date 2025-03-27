from typing import Any, List, Union

from prediction_market_agent_tooling.gtypes import ChecksumAddress
from pydantic import BaseModel

from prediction_market_agent.agents.safe_guard_agent.safe_api_models.transactions import (
    CreationTxInfo,
    CustomTxInfo,
    SettingsChangeTxInfo,
    SwapOrderTxInfo,
    TransferTxInfo,
)


class Sender(BaseModel):
    value: str
    name: Any
    logoUri: Any


class Recipient(BaseModel):
    value: str
    name: Any
    logoUri: Any


class TransferInfo(BaseModel):
    type: str
    value: str


class AddressInfo(BaseModel):
    value: str
    name: Any | None = None
    logoUri: Any | None = None


class To(BaseModel):
    value: ChecksumAddress
    name: Any | None = None
    logoUri: Any | None = None


class TxData(BaseModel):
    hexData: str | None = None
    dataDecoded: Any
    to: To
    value: str
    operation: int
    trustedDelegateCallTarget: Any | None = None
    addressInfoIndex: Any | None = None


class RefundReceiver(BaseModel):
    value: ChecksumAddress
    name: Any | None = None
    logoUri: Any | None = None


class Signer(BaseModel):
    value: str
    name: Any | None = None
    logoUri: Any | None = None


class Confirmation(BaseModel):
    signer: Signer
    signature: str
    submittedAt: int


class Proposer(BaseModel):
    value: str
    name: Any | None = None
    logoUri: Any | None = None


class DetailedExecutionInfo(BaseModel):
    type: str
    address: AddressInfo | None = None
    submittedAt: int | None = None
    nonce: int | None = None
    safeTxGas: str | None = None
    baseGas: str | None = None
    gasPrice: str | None = None
    gasToken: ChecksumAddress | None = None
    refundReceiver: RefundReceiver | None = None
    safeTxHash: str | None = None
    executor: Any | None = None
    signers: List[Signer] | None = None
    confirmationsRequired: int | None = None
    confirmations: List[Confirmation] | None = None
    rejectors: List[Any] | None = None
    gasTokenInfo: Any | None = None
    trusted: bool | None = None
    proposer: Proposer | None = None
    proposedByDelegate: Any | None = None


class DetailedTransactionResponse(BaseModel):
    safeAddress: ChecksumAddress
    txId: str
    executedAt: int | None = None
    txStatus: str
    txInfo: Union[
        CreationTxInfo,
        SettingsChangeTxInfo,
        TransferTxInfo,
        SwapOrderTxInfo,
        CustomTxInfo,
    ]
    txData: TxData | None = None
    txHash: str | None = None
    detailedExecutionInfo: DetailedExecutionInfo | None = None
    safeAppInfo: Any | None = None
    note: str | None = None
