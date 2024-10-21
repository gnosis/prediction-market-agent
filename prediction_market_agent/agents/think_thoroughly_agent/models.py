from eth_typing import HexAddress
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.markets.omen.data_models import OmenMarket
from pydantic import BaseModel


class PineconeMetadata(BaseModel):
    question_title: str
    market_address: HexAddress
    close_time_timestamp: int

    @staticmethod
    def from_omen_market(market: OmenMarket) -> "PineconeMetadata":
        return PineconeMetadata(
            question_title=market.question_title,
            market_address=market.id,
            close_time_timestamp=market.close_time.timestamp(),
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
