from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.gtypes import Probability, USD
from prediction_market_agent_tooling.deploy.betting_strategy import (
    BettingStrategy,
    MaxAccuracyWithKellyScaledBetsStrategy,
)
from prediction_market_agent.agents.utils import get_maximum_possible_bet_amount


class NoBiasAgent(DeployableTraderAgent):
    bet_on_n_markets_per_run = 5

    DEFAULT_P_YES = 0.12
    CONFIDENCE = 0.55

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        market_prob = float(market.p_yes)

        print(f"\nAnalyzing: {market.question}")
        print(f"Market p_yes: {market_prob:.3f}")
        print(f"Agent p_yes:  {self.DEFAULT_P_YES:.3f}")

        return ProbabilisticAnswer(
            p_yes=Probability(self.DEFAULT_P_YES),
            confidence=self.CONFIDENCE,
            reasoning="Simple NO-biased baseline using historical YES base rate.",
        )

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        return MaxAccuracyWithKellyScaledBetsStrategy(
            max_position_amount=get_maximum_possible_bet_amount(
                min_=USD(0.01),
                max_=USD(0.05),
                trading_balance=market.get_trade_balance(self.api_keys),
            ),
        )