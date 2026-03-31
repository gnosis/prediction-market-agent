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


class ContrarianAgent(DeployableTraderAgent):

    bet_on_n_markets_per_run = 4

    NO_THRESHOLD = 0.40
    CONFIDENCE = 0.55

    def answer_binary_market(self, market: AgentMarket):

        title = market.question
        market_prob = float(market.p_yes)

        print(f"\nAnalyzing: {title}")
        print(f"Market p_yes: {market_prob:.3f}")

        # Only bet when YES looks overpriced
        if market_prob < self.NO_THRESHOLD:
            print("Skipping: YES not overpriced enough.")
            return None

        print("Betting NO")

        return ProbabilisticAnswer(
            p_yes=Probability(1 - market_prob),
            confidence=self.CONFIDENCE,
            reasoning="NO-default baseline agent",
        )

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        return MaxAccuracyWithKellyScaledBetsStrategy(
            max_position_amount=get_maximum_possible_bet_amount(
                min_=USD(0.01),
                max_=USD(0.05),
                trading_balance=market.get_trade_balance(self.api_keys),
            ),
        )
