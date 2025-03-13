from prediction_market_agent_tooling.gtypes import ChecksumAddress
from pydantic import BaseModel


class TokenInfo(BaseModel):
    type: str
    address: ChecksumAddress
    decimals: int
    symbol: str
    name: str
    logoUri: str | None


class Item(BaseModel):
    tokenInfo: TokenInfo
    balance: str
    fiatBalance: str
    fiatConversion: str


class Balances(BaseModel):
    fiatTotal: str
    items: list[Item]
