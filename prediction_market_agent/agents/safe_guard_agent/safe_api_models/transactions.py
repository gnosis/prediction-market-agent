from typing import Any, List, Literal, Union

from prediction_market_agent_tooling.gtypes import ChecksumAddress
from pydantic import BaseModel


class Token(BaseModel):
    address: ChecksumAddress
    decimals: int
    logoUri: str | None = None
    name: str | None = None
    symbol: str | None = None
    trusted: bool | None = None
    imitation: bool | None = None


class Address(BaseModel):
    value: ChecksumAddress
    name: str | None = None
    logoUri: str | None = None


class TransferInfo(BaseModel):
    type: str
    value: str | None = None
    tokenAddress: str | None = None
    tokenId: str | None = None
    tokenName: str | None = None
    tokenSymbol: str | None = None
    decimals: int | None = None
    logoUri: str | None = None
    trusted: bool | None = None
    imitation: bool | None = None


class TransferTxInfo(BaseModel):
    type: Literal["Transfer"]
    humanDescription: str | None = None
    sender: Address
    recipient: Address
    direction: str
    transferInfo: TransferInfo


class CustomTxInfo(BaseModel):
    type: Literal["Custom"]
    humanDescription: str | None = None
    to: Address
    dataSize: str
    value: str
    methodName: str | None = None
    actionCount: int | None = None
    isCancellation: bool


class SettingsChangeTxInfo(BaseModel):
    type: Literal["SettingsChange"]
    humanDescription: str | None = None


class SwapOrderTxInfo(BaseModel):
    type: Literal["SwapOrder"]
    uid: str
    humanDescription: str | None = None
    status: str
    kind: str
    sellAmount: str
    buyAmount: str
    sellToken: Token
    buyToken: Token
    explorerUrl: str
    receiver: ChecksumAddress
    owner: ChecksumAddress


class CreationTxInfo(BaseModel):
    type: Literal["Creation"]
    humanDescription: str | None = None
    creator: Address
    transactionHash: str
    implementation: Address
    factory: Address
    saltNonce: str


class ModuleExecutionInfo(BaseModel):
    type: Literal["MODULE"]
    address: Address


class MultisigExecutionInfo(BaseModel):
    type: Literal["MULTISIG"]
    nonce: int
    confirmationsRequired: int
    confirmationsSubmitted: int
    missingSigners: List[Address] | None = None


class Transaction(BaseModel):
    txInfo: Union[
        CreationTxInfo,
        SettingsChangeTxInfo,
        TransferTxInfo,
        SwapOrderTxInfo,
        CustomTxInfo,
    ]
    id: str
    timestamp: int
    txStatus: str
    executionInfo: Union[ModuleExecutionInfo, MultisigExecutionInfo] | None = None
    safeAppInfo: Any | None = None
    txHash: str | None = None


class TransactionWithMultiSig(Transaction):
    executionInfo: MultisigExecutionInfo

    @staticmethod
    def from_tx(tx: Transaction) -> "TransactionWithMultiSig":
        return TransactionWithMultiSig.model_validate(tx.model_dump())


class TransactionResult(BaseModel):
    type: Literal["LABEL", "DATE_LABEL", "TRANSACTION", "CONFLICT_HEADER"]
    label: str | None = None
    transaction: Transaction | None = None
    conflictType: str | None = None


class TransactionResponse(BaseModel):
    count: int
    next: str | None = None
    previous: str | None = None
    results: List[TransactionResult]
