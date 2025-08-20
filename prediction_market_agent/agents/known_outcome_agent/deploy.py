from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.deploy.betting_strategy import (
    BettingStrategy,
    FullBinaryKellyBettingStrategy,
)
from prediction_market_agent_tooling.gtypes import USD
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket

from prediction_market_agent.agents.known_outcome_agent.known_outcome_agent import (
    Result,
    get_known_outcome,
)
from prediction_market_agent.agents.utils import (
    get_maximum_possible_bet_amount,
    market_is_saturated,
)
from prediction_market_agent.utils import APIKeys


class DeployableKnownOutcomeAgent(DeployableTraderAgent):
    model = "gpt-4-1106-preview"
    min_liquidity = USD(5)
    bet_on_n_markets_per_run: int = 2
    supported_markets = [MarketType.OMEN]

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        return FullBinaryKellyBettingStrategy(
            max_position_amount=get_maximum_possible_bet_amount(
                min_=USD(1),
                max_=USD(2),
                trading_balance=market.get_trade_balance(APIKeys()),
            ),
            max_price_impact=0.6,
        )

    def load(self) -> None:
        self.markets_with_known_outcomes: dict[str, Result] = {}

    def verify_market(self, market_type: MarketType, market: AgentMarket) -> bool:
        assert isinstance(
            market, OmenAgentMarket
        ), "It's true thanks to supported_markets property, this just makes mypy happy."

        # Assume very high probability markets are already known, and have
        # been correctly bet on, and therefore the value of betting on them
        # is low.
        if market_is_saturated(market=market):
            logger.info(
                f"Skipping market {market.url} with the question '{market.question}', because it is already saturated."
            )
            return False
        elif market.get_liquidity() < market.get_in_token(self.min_liquidity):
            logger.info(
                f"Skipping market {market.url} with the question '{market.question}', because it has insufficient liquidity (at least {self.min_liquidity} required)."
            )
            return False
        else:
            return True

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        try:
            outcome = get_known_outcome(
                model=self.model,
                question=market.question,
                max_tries=3,
            )
        except Exception as e:
            logger.error(
                f"Failed to predict market {market.url} with the question '{market.question}' because of '{e}'."
            )
            outcome = None
        if outcome and outcome.has_known_result():
            answer = ProbabilisticAnswer(
                p_yes=outcome.result.to_p_yes(),
                confidence=1.0,
                reasoning=outcome.reasoning,
            )
            logger.info(
                f"Picking market {market.url} with the question '{market.question}' with answer '{answer}'"
            )
            return answer

        logger.info(f"No definite answer found for the market {market.url}.")

        return None
