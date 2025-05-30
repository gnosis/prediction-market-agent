from typing import Any, List, Literal, Union

from prediction_market_agent_tooling.gtypes import ChecksumAddress
from pydantic import BaseModel, ConfigDict, Field


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

    def format_llm(self) -> str:
        return (
            f"Transaction type: {self.type} | "
            + (
                f"Human description: {self.humanDescription} | "
                if self.humanDescription
                else ""
            )
            + f"Sender: {self.sender.value} | "
            + f"Recipient: {self.recipient.value} | "
            + f"Direction: {self.direction} | "
            + f"Transfer token type: {self.transferInfo.type} | "
            + (
                f"Transfer token address: {self.transferInfo.tokenAddress} | "
                if self.transferInfo.tokenAddress
                else ""
            )
            + (
                f"Transfer token symbol: {self.transferInfo.tokenSymbol} | "
                if self.transferInfo.tokenSymbol
                else ""
            )
            + f"Transfer value: {self.transferInfo.value} | "
        )


class SwapTransferTxInfo(BaseModel):
    type: Literal["SwapTransfer"]
    humanDescription: str | None = None
    sender: Address
    recipient: Address
    direction: str
    transferInfo: TransferInfo
    uid: str
    status: str
    kind: str
    sellAmount: str
    buyAmount: str
    sellToken: Token
    buyToken: Token
    explorerUrl: str
    receiver: ChecksumAddress
    owner: ChecksumAddress

    def format_llm(self) -> str:
        return (
            f"Transaction type: {self.type} | "
            + (
                f"Human description: {self.humanDescription} | "
                if self.humanDescription
                else ""
            )
            + f"Sender: {self.sender.value} | "
            + f"Recipient: {self.recipient.value} | "
            + f"Direction: {self.direction} | "
            + f"Transfer token type: {self.transferInfo.type} | "
            + (
                f"Transfer token address: {self.transferInfo.tokenAddress} | "
                if self.transferInfo.tokenAddress
                else ""
            )
            + (
                f"Transfer token symbol: {self.transferInfo.tokenSymbol} | "
                if self.transferInfo.tokenSymbol
                else ""
            )
            + (
                f"Transfer value: {self.transferInfo.value} | "
                if self.transferInfo.value is not None
                else ""
            )
            + f"Status: {self.status} | "
            + f"Kind: {self.kind} | "
            + f"Sell amount: {self.sellAmount} | "
            + f"Buy amount: {self.buyAmount} | "
            + f"Sell token address: {self.sellToken.address} | "
            + (
                f"Sell token symbol: {self.sellToken.symbol} | "
                if self.sellToken.symbol
                else ""
            )
            + f"Buy token address: {self.buyToken.address} | "
            + (
                f"Buy token symbol: {self.buyToken.symbol} | "
                if self.buyToken.symbol
                else ""
            )
            + f"Receiver: {self.receiver} | "
            + f"Owner: {self.owner} | "
        )


class FullAppData(BaseModel):
    appCode: str
    environment: str
    metadata: dict[str, Any]
    version: str


class DurationOfPart(BaseModel):
    durationType: str


class StartTime(BaseModel):
    startType: str


class TwapOrderInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["TwapOrder"]
    humanDescription: str | None = None
    status: str
    kind: str
    class_: str = Field(alias="class")
    activeOrderUid: str | None = None
    validUntil: int
    sellAmount: str
    buyAmount: str
    executedSellAmount: str | None = None
    executedBuyAmount: str | None = None
    executedFee: str | None = None
    executedFeeToken: Token | None = None
    sellToken: Token
    buyToken: Token
    receiver: ChecksumAddress
    owner: ChecksumAddress
    fullAppData: FullAppData
    numberOfParts: str
    partSellAmount: str
    minPartLimit: str
    timeBetweenParts: int
    durationOfPart: DurationOfPart
    startTime: StartTime

    def format_llm(self) -> str:
        return (
            f"Transaction type: {self.type} | "
            + (
                f"Human description: {self.humanDescription} | "
                if self.humanDescription
                else ""
            )
            + f"Status: {self.status} | "
            + f"Kind: {self.kind} | "
            + f"Class: {self.class_} | "
            + (
                f"Active Order UID: {self.activeOrderUid} | "
                if self.activeOrderUid
                else ""
            )
            + f"Valid Until: {self.validUntil} | "
            + f"Sell amount: {self.sellAmount} | "
            + f"Buy amount: {self.buyAmount} | "
            + (
                f"Executed Sell Amount: {self.executedSellAmount} | "
                if self.executedSellAmount
                else ""
            )
            + (
                f"Executed Buy Amount: {self.executedBuyAmount} | "
                if self.executedBuyAmount
                else ""
            )
            + (f"Executed Fee: {self.executedFee} | " if self.executedFee else "")
            + (
                f"Executed Fee Token Address: {self.executedFeeToken.address} | "
                if self.executedFeeToken and self.executedFeeToken.address
                else ""
            )
            + (
                f"Executed Fee Token Symbol: {self.executedFeeToken.symbol} | "
                if self.executedFeeToken and self.executedFeeToken.symbol
                else ""
            )
            + f"Sell token address: {self.sellToken.address} | "
            + (
                f"Sell token symbol: {self.sellToken.symbol} | "
                if self.sellToken.symbol
                else ""
            )
            + f"Buy token address: {self.buyToken.address} | "
            + (
                f"Buy token symbol: {self.buyToken.symbol} | "
                if self.buyToken.symbol
                else ""
            )
            + f"Receiver: {self.receiver} | "
            + f"Owner: {self.owner} | "
            + f"Number of Parts: {self.numberOfParts} | "
            + f"Part Sell Amount: {self.partSellAmount} | "
            + f"Min Part Limit: {self.minPartLimit} | "
            + f"Time Between Parts: {self.timeBetweenParts} | "
            + f"Duration of Part: {self.durationOfPart} | "
            + f"Start Time: {self.startTime} | "
        )


class CustomTxInfo(BaseModel):
    type: Literal["Custom"]
    humanDescription: str | None = None
    to: Address
    dataSize: str
    value: str
    methodName: str | None = None
    actionCount: int | None = None
    isCancellation: bool

    def format_llm(self) -> str:
        return (
            (
                f"Human description: {self.humanDescription} | "
                if self.humanDescription
                else ""
            )
            + f"To address: {self.to.value} | "
            + f"Value: {self.value} | "
            + (f"Method name: {self.methodName} | " if self.methodName else "")
            + (
                f"Action count: {self.actionCount} | "
                if self.actionCount is not None
                else ""
            )
            + f"Is cancellation tx: {self.isCancellation} | "
        )


class SettingsChangeTxInfo(BaseModel):
    type: Literal["SettingsChange"]
    humanDescription: str | None = None

    def format_llm(self) -> str:
        return f"Transaction type: {self.type} | " + (
            f"Human description: {self.humanDescription} | "
            if self.humanDescription
            else ""
        )


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

    def format_llm(self) -> str:
        return (
            f"Transaction type: {self.type} | "
            + (
                f"Human description: {self.humanDescription} | "
                if self.humanDescription
                else ""
            )
            + f"Owner: {self.owner} | "
            + f"Recipient: {self.receiver} | "
            + f"Sell token address: {self.sellToken.address} | "
            + f"Sell token symbol: {self.sellToken.symbol} | "
            + f"Buy token address: {self.buyToken.address} | "
            + f"Buy token symbol: {self.buyToken.symbol} | "
            + f"Transfer value: {self.sellAmount} | "
        )


class CreationTxInfo(BaseModel):
    type: Literal["Creation"]
    humanDescription: str | None = None
    creator: Address
    transactionHash: str
    implementation: Address
    factory: Address
    saltNonce: str

    def format_llm(self) -> str:
        return f"Creator address: {self.creator.value} | " + (
            f"Human description: {self.humanDescription} | "
            if self.humanDescription
            else ""
        )


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
        SwapTransferTxInfo,
        TwapOrderInfo,
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
