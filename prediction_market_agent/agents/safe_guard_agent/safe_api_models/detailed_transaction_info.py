from typing import Any, List

from prediction_market_agent_tooling.gtypes import ChecksumAddress
from pydantic import BaseModel

from prediction_market_agent.agents.safe_guard_agent.safe_api_models.transactions import (
    SwapOrderTxInfo,
    TransferTxInfo,
    SettingsChangeTxInfo,
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


class To(BaseModel):
    value: ChecksumAddress
    name: Any
    logoUri: Any


class TxData(BaseModel):
    hexData: Any
    dataDecoded: Any
    to: To
    value: str
    operation: int
    trustedDelegateCallTarget: Any
    addressInfoIndex: Any


class RefundReceiver(BaseModel):
    value: ChecksumAddress
    name: Any
    logoUri: Any


class Signer(BaseModel):
    value: str
    name: Any
    logoUri: Any


class Signer1(BaseModel):
    value: str
    name: Any
    logoUri: Any


class Confirmation(BaseModel):
    signer: Signer1
    signature: str
    submittedAt: int


class Proposer(BaseModel):
    value: str
    name: Any
    logoUri: Any


class DetailedExecutionInfo(BaseModel):
    type: str
    submittedAt: int
    nonce: int
    safeTxGas: str
    baseGas: str
    gasPrice: str
    gasToken: ChecksumAddress
    refundReceiver: RefundReceiver
    safeTxHash: str
    executor: Any
    signers: List[Signer]
    confirmationsRequired: int
    confirmations: List[Confirmation]
    rejectors: List[Any]
    gasTokenInfo: Any
    trusted: bool
    proposer: Proposer
    proposedByDelegate: Any


class DetailedTransactionResponse(BaseModel):
    safeAddress: ChecksumAddress
    txId: str
    executedAt: int | None
    txStatus: str
    txInfo: TransferTxInfo | SwapOrderTxInfo | SettingsChangeTxInfo
    txData: TxData | None
    txHash: str | None
    detailedExecutionInfo: DetailedExecutionInfo | None
    safeAppInfo: Any
    note: str | None
