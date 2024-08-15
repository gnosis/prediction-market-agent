import random
import typing as t

from prediction_market_agent_tooling.deploy.agent import (
    Answer,
    DeployableTraderAgent,
    Probability,
)
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.markets import MarketType


class DeployableCoinFlipAgent(DeployableTraderAgent):
    def verify_market(self, market_type: MarketType, market: AgentMarket) -> bool:
        return True

    def answer_binary_market(self, market: AgentMarket) -> Answer | None:
        decision = random.choice([True, False])
        return Answer(
            decision=decision,
            confidence=0.5,
            p_yes=Probability(float(decision)),
            reasoning="I flipped a coin to decide.",
        )
