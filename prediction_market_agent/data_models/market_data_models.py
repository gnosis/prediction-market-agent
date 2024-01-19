from pydantic import BaseModel
from web3 import Web3
from web3.types import Wei
from prediction_market_agent.tools.types import USD, HexAddress, ChecksumAddress


class Market(BaseModel):
    id: HexAddress
    title: str
    collateralVolume: Wei
    usdVolume: USD
    collateralToken: HexAddress
    outcomes: list[str]

    @property
    def market_maker_contract_address(self) -> HexAddress:
        return self.id

    @property
    def market_maker_contract_address_checksummed(self) -> ChecksumAddress:
        return Web3.to_checksum_address(self.market_maker_contract_address)

    @property
    def collateral_token_contract_address(self) -> HexAddress:
        return self.collateralToken

    @property
    def collateral_token_contract_address_checksummed(self) -> ChecksumAddress:
        return Web3.to_checksum_address(self.collateral_token_contract_address)

    def __repr__(self):
        return f"Market: {self.title}"
