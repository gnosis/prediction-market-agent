import random
from datetime import timedelta

from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.deploy.trade_interval import (
    FixedInterval,
    TradeInterval,
)
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.markets.agent_market import AgentMarket, SortBy
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.markets.markets import MarketType


class DeployableCoinFlipAgent(DeployableTraderAgent):
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
    bet_on_n_markets_per_run = 2
    same_market_trade_interval: TradeInterval = FixedInterval(timedelta(days=14))
