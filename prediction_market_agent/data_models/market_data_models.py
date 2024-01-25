import typing as t
from pydantic import BaseModel
from web3 import Web3
from web3.types import Wei
from prediction_market_agent.tools.gtypes import (
    USD,
    HexAddress,
    ChecksumAddress,
    Probability,
    Mana,
    OmenOutcomeToken,
    xDai,
)
from datetime import datetime


@t.runtime_checkable
class MarketProtocol(t.Protocol):
    """
    Protocol for markets, which are questions that can be answered by the agent.

    Market must have an unique ID (so the agent can store information about it) and question to be answered.

    Protocol, instead of parent Base class, is used to allow markets with different data models to be used,
    for example, Omen doesn't have `question` field, but property `question` is created (see below) to adhere to this protocol.
    """

    @property
    def BET_AMOUNT_CURRENCY(self) -> str:
        ...

    @property
    def id(self) -> str:
        ...

    @property
    def question(self) -> str:
        ...


class Market(BaseModel):
    """
    Most probably this class will be empty and serve just as parent for markets,
    as we can't guarantee any common fields/methods between different markets,
    but, it's still good to have.
    """


class OmenMarket(Market):
    BET_AMOUNT_CURRENCY: t.ClassVar[str] = "xDai"

    id: HexAddress
    title: str
    collateralVolume: Wei
    usdVolume: USD
    collateralToken: HexAddress
    outcomes: list[str]
    outcomeTokenAmounts: list[OmenOutcomeToken]
    outcomeTokenMarginalPrices: t.Optional[list[xDai]]
    fee: t.Optional[Wei]

    @property
    def question(self) -> str:
        return self.title

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

    @property
    def outcomeTokenProbabilities(self) -> t.Optional[list[Probability]]:
        return (
            [Probability(float(x)) for x in self.outcomeTokenMarginalPrices]
            if self.outcomeTokenMarginalPrices is not None
            else None
        )

    def get_outcome_index(self, outcome: str) -> int:
        try:
            return self.outcomes.index(outcome)
        except ValueError:
            raise ValueError(f"Outcome `{outcome}` not found in `{self.outcomes}`.")

    def get_outcome_str(self, outcome_index: int) -> str:
        n_outcomes = len(self.outcomes)
        if outcome_index >= n_outcomes:
            raise ValueError(
                f"Outcome index `{outcome_index}` not valid. There are only "
                f"`{n_outcomes}` outcomes."
            )
        else:
            return self.outcomes[outcome_index]

    def __repr__(self) -> str:
        return f"Market: {self.title}"


class ManifoldPool(BaseModel):
    NO: float
    YES: float


class ManifoldMarket(Market):
    BET_AMOUNT_CURRENCY: t.ClassVar[str] = "Mana"

    id: str
    question: str
    creatorId: str
    closeTime: datetime
    createdTime: datetime
    creatorAvatarUrl: str
    creatorName: str
    creatorUsername: str
    isResolved: bool
    lastBetTime: datetime
    lastCommentTime: datetime
    lastUpdatedTime: datetime
    mechanism: str
    outcomeType: str
    p: float
    pool: ManifoldPool
    probability: Probability
    slug: str
    totalLiquidity: Mana
    uniqueBettorCount: int
    url: str
    volume: Mana
    volume24Hours: Mana

    @property
    def outcomes(self) -> list[str]:
        return list(self.pool.model_fields.keys())
