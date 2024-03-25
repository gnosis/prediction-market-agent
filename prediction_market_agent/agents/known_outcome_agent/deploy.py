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
from prediction_market_agent_tooling.tools.web3_utils import verify_address

from prediction_market_agent.agents.known_outcome_agent.known_outcome_agent import (
    Result,
    get_known_outcome,
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
            # Assume very high probability markets are already known, and have
            # been correctly bet on, and therefore the value of betting on them
            # is low.
            if not market_is_saturated(market=market):
                print(f"Checking market {market.id=} {market.question=}")
                answer = get_known_outcome(
                    model=self.model,
                    question=market.question,
                    max_tries=3,
                )
                if answer.has_known_outcome():
                    print(
                        f"Picking market {market.id=} {market.question=} with answer {answer.result=}"
                    )
                    picked_markets.append(market)
                    self.markets_with_known_outcomes[market.id] = answer.result

            else:
                print(
                    f"Skipping market {market.id=} {market.question=}, because it is already saturated."
                )

        return picked_markets

    def answer_binary_market(self, market: AgentMarket) -> bool:
        # The answer has already been determined in `pick_markets` so we just
        # return it here.
        return self.markets_with_known_outcomes[market.id].to_boolean()

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
            BET_FROM_ADDRESS=verify_address(
                "0xb611A9f02B318339049264c7a66ac3401281cc3c"
            ),
            BET_FROM_PRIVATE_KEY=private_key_type("EVAN_OMEN_BETTER_0_PKEY:latest"),
            OPENAI_API_KEY=SecretStr("EVAN_OPENAI_API_KEY:latest"),
            MANIFOLD_API_KEY=None,
        ),
        cron_schedule="0 */12 * * *",
        timeout=540,
    )
