from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.markets.omen.data_models import OmenMarket
from pydantic import BaseModel


class PineconeMetadata(BaseModel):
    question_title: str
    market_address: ChecksumAddress

    @staticmethod
    def from_omen_market(market: OmenMarket):
        return PineconeMetadata(
            question_title=market.question_title, market_address=market.id
        )


class CorrelatedMarketInput(BaseModel):
    current_p_yes: Probability
    question_title: str

    @staticmethod
    def from_omen_market(omen_market: OmenMarket) -> "CorrelatedMarketInput":
        return CorrelatedMarketInput(
            current_p_yes=omen_market.current_p_yes,
            question_title=omen_market.question_title,
        )
