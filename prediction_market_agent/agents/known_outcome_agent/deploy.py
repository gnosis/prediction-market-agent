import random
import typing as t

from prediction_market_agent_tooling.deploy.agent import Answer, DeployableAgent
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import BetAmount
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.tools.utils import check_not_none

from prediction_market_agent.agents.known_outcome_agent.known_outcome_agent import (
    Result,
    get_known_outcome,
)
from prediction_market_agent.agents.utils import market_is_saturated


class DeployableKnownOutcomeAgent(DeployableAgent):
    model = "gpt-4-1106-preview"
    min_liquidity = 5

    def load(self) -> None:
        self.markets_with_known_outcomes: dict[str, Result] = {}

    def pick_markets(self, markets: t.Sequence[AgentMarket]) -> list[AgentMarket]:
        picked_markets: list[AgentMarket] = []
        for market in markets:
            if not isinstance(market, OmenAgentMarket):
                raise NotImplementedError(
                    "This agent only supports predictions on Omen markets"
                )

            # Assume very high probability markets are already known, and have
            # been correctly bet on, and therefore the value of betting on them
            # is low.
            if market_is_saturated(market=market):
                logger.info(
                    f"Skipping market {market.url} with the question '{market.question}', because it is already saturated."
                )
            elif market.get_liquidity_in_xdai() < self.min_liquidity:
                logger.info(
                    f"Skipping market {market.url} with the question '{market.question}', because it has insufficient liquidity (at least {self.min_liquidity} required)."
                )
            else:
                picked_markets.append(market)

        # If all markets have a closing time set, pick the earliest closing.
        # Otherwise pick randomly.
        N_TO_PICK = 5
        if all(market.close_time for market in picked_markets):
            picked_markets = sorted(
                picked_markets, key=lambda market: check_not_none(market.close_time)
            )[:N_TO_PICK]
        else:
            picked_markets = random.sample(
                picked_markets, min(len(picked_markets), N_TO_PICK)
            )
        return picked_markets

    def answer_binary_market(self, market: AgentMarket) -> Answer | None:
        try:
            outcome = get_known_outcome(
                model=self.model,
                question=market.question,
                max_tries=3,
                callbacks=[self.langfuse_wrapper.get_langfuse_handler()],
            )
        except Exception as e:
            logger.error(
                f"Failed to predict market {market.url} with the question '{market.question}' because of '{e}'."
            )
            outcome = None
        if outcome and outcome.has_known_result():
            answer = Answer(
                decision=outcome.result.to_boolean(),
                p_yes=outcome.result.to_p_yes(),
                confidence=1.0,
            )
            logger.info(
                f"Picking market {market.url} with the question '{market.question}' with answer '{answer}'"
            )
            return answer

        logger.info(f"No definite answer found for the market {market.url}.")

        return None

    def calculate_bet_amount(self, answer: Answer, market: AgentMarket) -> BetAmount:
        if isinstance(market, OmenAgentMarket):
            return BetAmount(amount=1.0, currency=market.currency)
        else:
            raise NotImplementedError("This agent only supports xDai markets")
