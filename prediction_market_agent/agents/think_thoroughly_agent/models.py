from eth_typing import HexAddress
from prediction_market_agent_tooling.deploy.constants import (
    YES_OUTCOME_LOWERCASE_IDENTIFIER,
    NO_OUTCOME_LOWERCASE_IDENTIFIER,
)
from prediction_market_agent_tooling.gtypes import Probability, OutcomeStr
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
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
            close_time_timestamp=int(market.close_time.timestamp()),
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


class ThinkThoroughlyPrediction(BaseModel):
    reasoning: str
    p_yes: float
    p_no: float
    confidence: float

    def to_probabilistic_answer(self) -> ProbabilisticAnswer:
        return ProbabilisticAnswer(
            probabilities={
                OutcomeStr(YES_OUTCOME_LOWERCASE_IDENTIFIER.capitalize()): self.p_yes,
                OutcomeStr(NO_OUTCOME_LOWERCASE_IDENTIFIER.capitalize()): self.p_no,
            },
            confidence=self.confidence,
            reasoning=self.reasoning,
        )
