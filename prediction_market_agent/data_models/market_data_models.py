import typing as t
from pydantic import BaseModel
from web3 import Web3
from web3.types import Wei
from prediction_market_agent.markets.all_markets import Currency
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


class AgentMarket(BaseModel):
    """
    Common market class that can be created from vendor specific markets.
    Contains everything that is needed for an agent to make a prediction.
    """

    id: str
    question: str
    outcomes: list[str]
    bet_amount_currency: Currency
    original_market: t.Union["OmenMarket", "ManifoldMarket"]


class OmenMarket(BaseModel):
    """
    https://aiomen.eth.limo
    """

    BET_AMOUNT_CURRENCY: Currency = Currency.xDai

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

    def to_agent_market(self) -> AgentMarket:
        return AgentMarket(
            id=self.id,
            question=self.title,
            outcomes=self.outcomes,
            bet_amount_currency=self.BET_AMOUNT_CURRENCY,
            original_market=self,
        )

    def __repr__(self) -> str:
        return f"Omen's market: {self.title}"


class ManifoldPool(BaseModel):
    NO: float
    YES: float


class ManifoldMarket(BaseModel):
    """
    https://manifold.markets
    """

    BET_AMOUNT_CURRENCY: Currency = Currency.Mana

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

    def to_agent_market(self) -> "AgentMarket":
        return AgentMarket(
            id=self.id,
            question=self.question,
            outcomes=self.outcomes,
            bet_amount_currency=self.BET_AMOUNT_CURRENCY,
            original_market=self,
        )

    def __repr__(self) -> str:
        return f"Manifold's market: {self.question}"
