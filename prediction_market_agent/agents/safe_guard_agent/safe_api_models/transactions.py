from typing import Any, List, Literal, Optional

from prediction_market_agent_tooling.gtypes import ChecksumAddress
from pydantic import BaseModel


class Token(BaseModel):
    address: ChecksumAddress
    decimals: int
    logoUri: str
    name: str
    symbol: str
    trusted: bool


class Sender(BaseModel):
    value: str
    name: str | None
    logoUri: str | None


class Recipient(BaseModel):
    value: str
    name: str | None
    logoUri: str | None


class TransferInfo(BaseModel):
    type: str
    value: str


class TransferTxInfo(BaseModel):
    type: Literal["Transfer"]
    humanDescription: str | None
    sender: Sender
    recipient: Recipient
    direction: str
    transferInfo: TransferInfo


class CancellationTxInfo(BaseModel):
    type: Literal["Custom"]
    humanDescription: str | None
    to: Recipient
    dataSize: str
    value: str
    methodName: str | None
    actionCount: int | None
    isCancellation: Literal[True]


class SwapOrderTxInfo(BaseModel):
    type: Literal["SwapOrder"]
    uid: str
    humanDescription: str | None
    status: str
    kind: str
    sellAmount: str
    buyAmount: str
    sellToken: Token
    buyToken: Token
    explorerUrl: str
    receiver: ChecksumAddress
    owner: ChecksumAddress


class Creator(BaseModel):
    value: str
    name: str | None
    logoUri: str | None


class Implementation(BaseModel):
    value: str
    name: str
    logoUri: str


class Factory(BaseModel):
    value: str
    name: str
    logoUri: str


class CreationTxInfo(BaseModel):
    type: Literal["Creation"]
    humanDescription: Any
    creator: Creator
    transactionHash: str
    implementation: Implementation
    factory: Factory
    saltNonce: str


class MissingSigner(BaseModel):
    value: str
    name: str | None
    logoUri: str | None


class ExecutionInfo(BaseModel):
    type: str
    nonce: int
    confirmationsRequired: int
    confirmationsSubmitted: int
    missingSigners: List[MissingSigner] | None


class Transaction(BaseModel):
    txInfo: CreationTxInfo | TransferTxInfo | SwapOrderTxInfo | CancellationTxInfo
    id: str
    timestamp: int
    txStatus: str
    executionInfo: ExecutionInfo | None
    safeAppInfo: Any
    txHash: Any


class TransactionResult(BaseModel):
    type: Literal["LABEL", "DATE_LABEL", "TRANSACTION", "CONFLICT_HEADER"]
    label: Optional[str] = None
    transaction: Optional[Transaction] = None
    conflictType: Optional[str] = None


class TransactionResponse(BaseModel):
    count: int
    next: Any
    previous: Any
    results: List[TransactionResult]
