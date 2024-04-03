import getpass
from decimal import Decimal

from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.deploy.constants import OWNER_KEY
from prediction_market_agent_tooling.gtypes import SecretStr, private_key_type
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import BetAmount, Currency
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.utils import (
    get_current_git_commit_sha,
    get_current_git_url,
)

from prediction_market_agent.agents.known_outcome_agent.known_outcome_agent import (
    Result,
    get_known_outcome,
    has_question_event_happened_in_the_past,
)


def market_is_saturated(market: AgentMarket) -> bool:
    return market.p_yes > 0.95 or market.p_no > 0.95


class DeployableKnownOutcomeAgent(DeployableAgent):
    model = "gpt-4-1106-preview"

    def load(self) -> None:
        self.markets_with_known_outcomes: dict[str, Result] = {}

    def pick_markets(self, markets: list[AgentMarket]) -> list[AgentMarket]:
        picked_markets: list[AgentMarket] = []
        for market in markets:
            print(f"Looking at market {market.id=} {market.question=}")

            # Assume very high probability markets are already known, and have
            # been correctly bet on, and therefore the value of betting on them
            # is low.
            if market_is_saturated(market=market):
                print(
                    f"Skipping market {market.id=} {market.question=}, because it is already saturated."
                )
                continue

            # TODO it is currently expensive and slow to run the full evaluation
            # on all markets, so we only run it on markets that address events
            # that have already happened in the past.
            if has_question_event_happened_in_the_past(
                model=self.model, question=market.question
            ):
                print(f"Picking market {market.id=} {market.question=}")
                picked_markets.append(market)

        return picked_markets

    def answer_binary_market(self, market: AgentMarket) -> bool | None:
        try:
            answer = get_known_outcome(
                model=self.model,
                question=market.question,
                max_tries=3,
            )
        except Exception as e:
            print(
                f"Error: Failed to predict market {market.id=} {market.question=}: {e}"
            )
            answer = None
        if answer and answer.has_known_outcome():
            print(
                f"Picking market {market.id=} {market.question=} with answer {answer.result=}"
            )
            return answer.result.to_boolean()

        return None

    def calculate_bet_amount(self, answer: bool, market: AgentMarket) -> BetAmount:
        if market.currency == Currency.xDai:
            return BetAmount(amount=Decimal(0.1), currency=Currency.xDai)
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
