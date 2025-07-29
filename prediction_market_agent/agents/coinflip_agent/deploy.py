import random

from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.markets.agent_market import AgentMarket, SortBy
from prediction_market_agent_tooling.markets.data_models import (
    ProbabilisticAnswer,
    ScalarProbabilisticAnswer,
)
from prediction_market_agent_tooling.markets.markets import MarketType


class DeployableCoinFlipAgent(DeployableTraderAgent):
    n_markets_to_fetch = 10

    def verify_market(self, market_type: MarketType, market: AgentMarket) -> bool:
        return True

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        decision = random.choice([True, False])
        return ProbabilisticAnswer(
            confidence=0.5,
            p_yes=Probability(float(decision)),
            reasoning="I flipped a coin to decide.",
        )


class DeployableCoinFlipAgentByHighestLiquidity(DeployableCoinFlipAgent):
    get_markets_sort_by = SortBy.HIGHEST_LIQUIDITY
    bet_on_n_markets_per_run = 1
    n_markets_to_fetch = 2
    # same_market_trade_interval: TradeInterval = FixedInterval(timedelta(days=14))

    def answer_scalar_market(
        self, market: AgentMarket
    ) -> ScalarProbabilisticAnswer | None:
        return ScalarProbabilisticAnswer(
            confidence=0.5,
            upperBound=market.upper_bound,
            lowerBound=market.lower_bound,
            scalar_value=(market.upper_bound + market.lower_bound) / 2,
            reasoning="I flipped a coin to decide.",
        )
