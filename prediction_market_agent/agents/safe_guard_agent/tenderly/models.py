from typing import Union, Optional, Literal, List

from pydantic import BaseModel


class TokenInfo(BaseModel):
    standard: Literal["ERC20", "ERC721", "NativeCurrency"]
    type: Literal["Fungible", "NonFungible", "Native"]
    contract_address: Optional[str]
    symbol: Optional[str]
    name: Optional[str]
    logo: Optional[str]
    decimals: Optional[int]
    dollar_value: Optional[float]


class AssetChange(BaseModel):
    type: Literal["Transfer", "Mint", "Burn"]
    from_: Optional[str] = None  # 'from' is a keyword in Python
    to: Optional[str] = None
    amount: Optional[str]
    raw_amount: Optional[str]
    dollar_value: Optional[str]
    token_info: TokenInfo
    token_id: Optional[str] = None


class SimulationAssetChanges(BaseModel):
    asset_changes: List[AssetChange]


class TransactionInfo(BaseModel):
    asset_changes: List[AssetChange]


class SimulateNftTransaction(BaseModel):
    transaction_info: TransactionInfo


class SimulateNftAssetChangesPostResponse(BaseModel):
    transaction: SimulateNftTransaction


class Transaction(BaseModel):
    hash: str
    block_hash: Optional[str]
    block_number: int
    from_: str  # 'from' is a keyword in Python
    gas: int
    gas_price: int
    gas_fee_cap: int
    gas_tip_cap: int
    cumulative_gas_used: int
    gas_used: int
    effective_gas_price: int
    input: str


class FullModeSimulatePostResponse(BaseModel):
    transaction: Transaction


class QuickModeSimulatePostResponse(BaseModel):
    transaction: Transaction


class AbiModeSimulatePostResponse(BaseModel):
    transaction: Transaction


class FailedTxSimulatePostResponse(BaseModel):
    transaction: Transaction


class SimulatePostResponse(BaseModel):
    __root__: Union[
        FullModeSimulatePostResponse,
        QuickModeSimulatePostResponse,
        AbiModeSimulatePostResponse,
        FailedTxSimulatePostResponse,
        SimulationAssetChanges,
        SimulateNftAssetChangesPostResponse,
    ]
