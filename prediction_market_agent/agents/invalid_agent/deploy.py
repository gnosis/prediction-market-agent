from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.deploy.betting_strategy import (
    BettingStrategy,
    KellyBettingStrategy,
)
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.is_invalid import is_invalid

from prediction_market_agent.agents.utils import get_maximum_possible_bet_amount
from prediction_market_agent.utils import APIKeys


class InvalidAgent(DeployableTraderAgent):
    """This agent works only on Omen.
    Because on Omen, after market is resolved as invalid, outcome tokens are worth equally, which means one can be profitable by buying the cheapest token.
    Also the function to mark invalid questions is based on Omen-resolution rules."""

    bet_on_n_markets_per_run: int = 10
    supported_markets = [MarketType.OMEN]

    def verify_market(self, market_type: MarketType, market: AgentMarket) -> bool:
        if self.have_bet_on_market_since(market, since=self.same_market_bet_interval):
            return False

        # If the market is new, don't bet on it as the potential profit from market invalidity is low.
        if 0.45 <= market.current_p_yes <= 0.55:
            return False

        # In contrast to the parent implementation, this agent will place bets only on invalid markets,
        # it doesn't care whether the market is predictable or not.
        return is_invalid(market.question)

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        # Keep Kelly here! See `answer_binary_market`.
        return KellyBettingStrategy(
            max_bet_amount=get_maximum_possible_bet_amount(
                min_=1, max_=5, trading_balance=market.get_trade_balance(APIKeys())
            )
        )

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        return ProbabilisticAnswer(
            confidence=1.0,
            p_yes=Probability(0.5),
            reasoning="This market has been assessed as invalid, thus a probability of 0.5 is predicted. The Kelly strategy will opt for the less expensive outcome, adjusting the probability to 0.5, which is anticipated to be the final resolution where all outcome tokens hold equal value.",
        )
