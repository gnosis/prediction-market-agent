# type: ignore
import getpass
import random
from decimal import Decimal

from loguru import logger
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.deploy.constants import OWNER_KEY
from prediction_market_agent_tooling.gtypes import SecretStr, private_key_type
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import BetAmount
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.tools.utils import (
    check_not_none,
    get_current_git_commit_sha,
    get_current_git_url,
)

from prediction_market_agent.agents.known_outcome_agent.known_outcome_agent import (
    Result,
    get_known_outcome,
)
from prediction_market_agent.agents.utils import market_is_saturated


class DeployableKnownOutcomeAgent(DeployableAgent):
    model = "gpt-4-turbo-preview"

    def load(self) -> None:
        self.markets_with_known_outcomes: dict[str, Result] = {}

    def pick_markets(self, markets: list[AgentMarket]) -> list[AgentMarket]:
        picked_markets: list[AgentMarket] = []
        for market in markets:
            if not isinstance(market, OmenAgentMarket):
                raise NotImplementedError(
                    "This agent only supports predictions on Omen markets"
                )

            logger.info(f"Looking at market {market.id=} {market.question=}")

            # Assume very high probability markets are already known, and have
            # been correctly bet on, and therefore the value of betting on them
            # is low.
            if market_is_saturated(market=market):
                logger.info(
                    f"Skipping market {market.id=} {market.question=}, because it is already saturated."
                )
            elif market.get_liquidity_in_xdai() < 5:
                logger.info(
                    f"Skipping market {market.id=} {market.question=}, because it has insufficient liquidity."
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

    def answer_binary_market(self, market: AgentMarket) -> bool | None:
        try:
            answer = get_known_outcome(
                model=self.model,
                question=market.question,
                max_tries=3,
            )
        except Exception as e:
            logger.error(
                f"Failed to predict market {market.id=} {market.question=}: {e}"
            )
            answer = None
        if answer and answer.has_known_result():
            logger.info(
                f"Picking market {market.id=} {market.question=} with answer {answer.result=}"
            )
            return answer.result.to_boolean()

        return None

    def calculate_bet_amount(self, answer: bool, market: AgentMarket) -> BetAmount:
        if isinstance(market, OmenAgentMarket):
            return BetAmount(amount=(Decimal(1.0)), currency=market.currency)
        else:
            raise NotImplementedError("This agent only supports xDai markets")


if __name__ == "__main__":
    agent = DeployableKnownOutcomeAgent()
    agent.deploy_gcp(
        repository=f"git+{get_current_git_url()}@{get_current_git_commit_sha()}",
        market_type=MarketType.OMEN,
        labels={OWNER_KEY: getpass.getuser()},
        secrets={
            "TAVILY_API_KEY": "GNOSIS_AI_TAVILY_API_KEY:latest",
        },
        memory=1024,
        api_keys=APIKeys(
            BET_FROM_PRIVATE_KEY=private_key_type("EVAN_OMEN_BETTER_0_PKEY:latest"),
            OPENAI_API_KEY=SecretStr("EVAN_OPENAI_API_KEY:latest"),
            MANIFOLD_API_KEY=None,
        ),
        cron_schedule="0 */12 * * *",
        timeout=540,
    )
