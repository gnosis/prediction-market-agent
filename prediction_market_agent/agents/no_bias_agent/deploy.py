import os
from dotenv import load_dotenv

from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.gtypes import Probability, USD
from prediction_market_agent_tooling.deploy.betting_strategy import (
    BettingStrategy,
    MaxAccuracyWithKellyScaledBetsStrategy,
)
from prediction_market_agent.agents.utils import get_maximum_possible_bet_amount

load_dotenv()


class NoBiasAgent(DeployableTraderAgent):
    bet_on_n_markets_per_run = 2

    # Historical finding from your resolved-market analysis
    HISTORICAL_YES_RATE = 0.15

    # Only bet if the market is pricing YES well above history
    MIN_EDGE = 0.15
    MIN_MARKET_PYES = 0.40

    # How strongly to trust the base rate vs. market price
    # 1.0 = use only historical base rate
    # lower values = allow some market information in
    BASE_RATE_WEIGHT = 0.85

    CONFIDENCE = 0.60

    def answer_binary_market(self, market: AgentMarket):
        title = market.question
        market_prob = float(market.p_yes)

        print(f"\nAnalyzing: {title}")
        print(f"Market p_yes: {market_prob:.3f}")

        # Only look at markets where YES is priced fairly high
        if market_prob < self.MIN_MARKET_PYES:
            print("Skipping: market YES probability not high enough.")
            return None

        # Compare market price to your historical YES base rate
        edge = market_prob - self.HISTORICAL_YES_RATE
        print(f"Historical YES rate: {self.HISTORICAL_YES_RATE:.3f}")
        print(f"Estimated overpricing edge: {edge:.3f}")

        if edge < self.MIN_EDGE:
            print("Skipping: estimated edge is too small.")
            return None

        # Blend historical base rate with current market price.
        # Since you want a NO-biased agent, this keeps p_yes low.
        adjusted_p_yes = (
            self.BASE_RATE_WEIGHT * self.HISTORICAL_YES_RATE
            + (1 - self.BASE_RATE_WEIGHT) * market_prob
        )

        # Safety cap so the forecast stays clearly NO-leaning
        adjusted_p_yes = min(adjusted_p_yes, 0.35)

        print(f"Adjusted p_yes: {adjusted_p_yes:.3f}")
        print("Betting NO")

        return ProbabilisticAnswer(
            p_yes=Probability(adjusted_p_yes),
            confidence=self.CONFIDENCE,
            reasoning=(
                f"NO-biased agent using historical YES base rate "
                f"({self.HISTORICAL_YES_RATE:.2f}) versus market YES probability "
                f"({market_prob:.2f})."
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