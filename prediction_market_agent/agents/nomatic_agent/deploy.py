import os
import typing as t
from collections import Counter
from datetime import timedelta

from dotenv import load_dotenv

from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.gtypes import Probability, USD
from prediction_market_agent_tooling.deploy.betting_strategy import (
    BettingStrategy,
    MaxAccuracyWithKellyScaledBetsStrategy,
)
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    FilterBy,
    OmenSubgraphHandler,
    SortBy,
)
from prediction_market_agent_tooling.tools.utils import check_not_none, utcnow
from prediction_market_agent.agents.utils import get_maximum_possible_bet_amount

load_dotenv()


class NoMaticAgent(DeployableTraderAgent):
    supported_markets = [MarketType.OMEN]
    bet_on_n_markets_per_run = 2

    LOOKBACK_DAYS = 30
    MIN_MARKET_PYES = 0.60
    MIN_EDGE = 0.20
    BASE_RATE_WEIGHT = 0.85
    MAX_ADJUSTED_PYES = 0.35
    CONFIDENCE_FLOOR = 0.55
    CONFIDENCE_CAP = 0.80
    MIN_RECENT_NO_RATE = 0.55
    MAX_CLOSE_DAYS = 14

    def load(self) -> None:
        start_date = utcnow() - timedelta(days=self.LOOKBACK_DAYS)

        recent_markets = OmenSubgraphHandler().get_omen_markets_simple(
            limit=None,
            filter_by=FilterBy.RESOLVED,
            sort_by=SortBy.NONE,
            include_categorical_markets=False,
            created_after=start_date,
        )

        resolutions = [
            m.question.boolean_outcome
            for m in recent_markets
            if m.question.boolean_outcome is not None
        ]

        if not resolutions:
            self.recent_yes_rate = 0.15
            self.recent_no_rate = 0.85
            print("No recent resolved markets found. Falling back to default base rates.")
            return

        counter = Counter(resolutions)
        total = sum(counter.values())

        self.recent_yes_rate = counter.get(True, 0) / total
        self.recent_no_rate = counter.get(False, 0) / total

        print(
            f"Loaded recent base rates from {total} resolved markets: "
            f"YES={self.recent_yes_rate:.3f}, NO={self.recent_no_rate:.3f}"
        )

    def get_markets(
        self,
        market_type: MarketType,
    ) -> t.Sequence[AgentMarket]:
        max_close_time = utcnow() + timedelta(days=self.MAX_CLOSE_DAYS)

        markets = super().get_markets(market_type)
        filtered_markets = [
            m for m in markets
            if check_not_none(m.close_time) < max_close_time
        ]
        return filtered_markets

    def answer_binary_market(self, market: AgentMarket):
        title = market.question
        market_prob = float(market.p_yes)

        print(f"\nAnalyzing: {title}")
        print(f"Market p_yes: {market_prob:.3f}")
        print(f"Recent YES base rate: {self.recent_yes_rate:.3f}")
        print(f"Recent NO base rate: {self.recent_no_rate:.3f}")

        if self.recent_no_rate < self.MIN_RECENT_NO_RATE:
            print("Skipping: recent market skew is not NO-dominant enough.")
            return None

        if market_prob < self.MIN_MARKET_PYES:
            print("Skipping: market YES probability not high enough.")
            return None

        edge = market_prob - self.recent_yes_rate
        print(f"Estimated overpricing edge: {edge:.3f}")

        if edge < self.MIN_EDGE:
            print("Skipping: estimated edge is too small.")
            return None

        adjusted_p_yes = (
            self.BASE_RATE_WEIGHT * self.recent_yes_rate
            + (1 - self.BASE_RATE_WEIGHT) * market_prob
        )
        adjusted_p_yes = min(adjusted_p_yes, self.MAX_ADJUSTED_PYES)

        confidence = min(self.CONFIDENCE_CAP, self.CONFIDENCE_FLOOR + edge)

        print(f"Adjusted p_yes: {adjusted_p_yes:.3f}")
        print(f"Confidence: {confidence:.3f}")
        print("Betting NO")

        return ProbabilisticAnswer(
            p_yes=Probability(adjusted_p_yes),
            confidence=confidence,
            reasoning=(
                f"NO-biased agent using recent YES base rate "
                f"({self.recent_yes_rate:.2f}) versus market YES probability "
                f"({market_prob:.2f}), with estimated edge {edge:.2f}."
            ),
        )

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        return MaxAccuracyWithKellyScaledBetsStrategy(
            max_position_amount=get_maximum_possible_bet_amount(
                min_=USD(0.01),
                max_=USD(0.05),
                trading_balance=market.get_trade_balance(self.api_keys),
            ),
        )