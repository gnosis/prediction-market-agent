from typing import Any, List, Union

from prediction_market_agent_tooling.gtypes import ChecksumAddress
from pydantic import BaseModel

from prediction_market_agent.agents.safe_guard_agent.safe_api_models.transactions import (
    CreationTxInfo,
    CustomTxInfo,
    SettingsChangeTxInfo,
    SwapOrderTxInfo,
    SwapTransferTxInfo,
    TransferTxInfo,
)


class AddressInfo(BaseModel):
    value: ChecksumAddress
    name: Any | None = None
    logoUri: Any | None = None


class TxData(BaseModel):
    hexData: str | None = None
    dataDecoded: dict[str, Any] | None = None
    to: AddressInfo
    value: str
    operation: int
    trustedDelegateCallTarget: Any | None = None
    addressInfoIndex: Any | None = None


class Confirmation(BaseModel):
    signer: AddressInfo
    signature: str
    submittedAt: int


class DetailedExecutionInfo(BaseModel):
    type: str
    address: AddressInfo | None = None
    submittedAt: int | None = None
    nonce: int | None = None
    safeTxGas: str | None = None
    baseGas: str | None = None
    gasPrice: str | None = None
    gasToken: ChecksumAddress | None = None
    refundReceiver: AddressInfo | None = None
    safeTxHash: str | None = None
    executor: AddressInfo | None = None
    signers: List[AddressInfo] | None = None
    confirmationsRequired: int | None = None
    confirmations: List[Confirmation] | None = None
    rejectors: List[Any] | None = None
    gasTokenInfo: Any | None = None
    trusted: bool | None = None
    proposer: AddressInfo | None = None
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
        SwapTransferTxInfo,
    ]
    txData: TxData | None = None
    txHash: str | None = None
    detailedExecutionInfo: DetailedExecutionInfo | None = None
    safeAppInfo: Any | None = None
    note: str | None = None
