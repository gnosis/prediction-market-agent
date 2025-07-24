import typing as t
from collections import Counter
from datetime import timedelta

from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.deploy.betting_strategy import (
    BettingStrategy,
    MultiCategoricalMaxAccuracyBettingStrategy,
)
from prediction_market_agent_tooling.deploy.trade_interval import (
    FixedInterval,
    TradeInterval,
)
from prediction_market_agent_tooling.gtypes import USD, Probability
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import AgentMarket, SortBy
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    FilterBy,
    OmenSubgraphHandler,
    SortBy,
)
from prediction_market_agent_tooling.tools.utils import check_not_none, utcnow


class SkewAgent(DeployableTraderAgent):
    supported_markets = [MarketType.OMEN]

    # Agent is as cheap as it gets, we can process all that we have funds for.
    n_markets_to_fetch = 1000
    bet_on_n_markets_per_run = 1000
    # Basically never re-bet on the market, because it's not doing any kind of research.
    same_market_trade_interval: TradeInterval = FixedInterval(timedelta(days=1995))
    # Process from newest, to take advantage of starting 0.5/0.5 prices.
    get_markets_sort_by = SortBy.NEWEST

    def load(self) -> None:
        # Find the majority resolution based on the last 30 days of markets.
        start_date = utcnow() - timedelta(days=30)

        recent_markets = OmenSubgraphHandler().get_omen_markets_simple(
            limit=None,
            filter_by=FilterBy.RESOLVED,
            sort_by=SortBy.NONE,
            include_categorical_markets=False,
            created_after=start_date,
        )
        resolutions = [m.question.boolean_outcome for m in recent_markets]

        counter = Counter(resolutions)
        dist = {c: v / sum(counter.values()) for c, v in counter.items()}

        self.majority_resolution = counter.most_common(1)[0][0]

        logger.info(
            f"Majority resolution is {self.majority_resolution}, distribution: {dist}, based on {len(recent_markets)} markets."
        )

        if dist[self.majority_resolution] < 0.55:
            raise ValueError(
                "If this happens, we can most probably shut this agent down and test again the agent from https://github.com/gnosis/prediction-market-agent/pull/932."
            )

    def get_markets(
        self,
        market_type: MarketType,
    ) -> t.Sequence[AgentMarket]:
        # Process only markets closing soon, to not have funds locked in markets that run for years.
        max_close_time = utcnow() + timedelta(days=14)

        markets = super().get_markets(market_type)
        filtered_markets = [
            m for m in markets if check_not_none(m.close_time) < max_close_time
        ]

        return filtered_markets

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        # Set trading balance in a way to have at least 100 trades, so we can relay on the statistics.
        # But the agent will stop processing if the bet can't be at least 0.01.
        max_position_amount = max(
            USD(0.01), market.get_trade_balance(self.api_keys) / 100
        )
        return MultiCategoricalMaxAccuracyBettingStrategy(
            max_position_amount=max_position_amount,
        )

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        return ProbabilisticAnswer(
            p_yes=Probability(float(self.majority_resolution)),
            confidence=1.0,
            reasoning="Chosen based on the majority resolution in the recent history.",
        )
