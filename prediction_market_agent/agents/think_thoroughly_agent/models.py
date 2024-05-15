from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.markets.omen.data_models import OmenMarket
from pydantic import BaseModel


class PineconeMetadata(BaseModel):
    question_title: str
    market_address: ChecksumAddress


class CorrelatedMarketInput(BaseModel):
    current_p_yes: Probability
    current_p_no: Probability
    question_title: str

    @staticmethod
    def from_omen_market(omen_market: OmenMarket) -> "CorrelatedMarketInput":
        return CorrelatedMarketInput(
            current_p_yes=omen_market.current_p_yes,
            current_p_no=omen_market.current_p_no,
            question_title=omen_market.question_title,
        )
