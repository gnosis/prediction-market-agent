from decimal import Decimal
import random
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import BetAmount, Currency
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.crewai_subsequential_agent.crewai_agent_subquestions import CrewAIAgentSubquestions
from prediction_market_agent.agents.known_outcome_agent.known_outcome_agent import (
    Result,
)


def market_is_saturated(market: AgentMarket) -> bool:
    return market.p_yes > 0.95 or market.p_no > 0.95


class DeployableThinkThoroughlyAgent(DeployableAgent):
    # For cheaper credits at this experimental stage
    model = "gpt-3.5-turbo"

    def load(self) -> None:
        self.markets_with_known_outcomes: dict[str, Result] = {}

    def pick_markets(self, markets: list[AgentMarket]) -> list[AgentMarket]:
        # We simply pick 5 random markets to bet on
        picked_markets: list[AgentMarket] = []
        random.shuffle(markets)
        for market in markets:
            # Assume very high probability markets are already known, and have
            # been correctly bet on, and therefore the value of betting on them
            # is low.
            if not market_is_saturated(market=market):
                picked_markets.append(market)
                if len(picked_markets) == 5:
                    break

        return picked_markets

    def answer_binary_market(self, market: AgentMarket) -> bool:
        # The answer has already been determined in `pick_markets` so we just
        # return it here.
        agent = CrewAIAgentSubquestions()
        result = agent.answer_binary_market(market)
        return result

    def calculate_bet_amount(self, answer: bool, market: AgentMarket) -> BetAmount:
        if market.currency == Currency.xDai:
            return BetAmount(amount=Decimal(0.1), currency=Currency.xDai)
        else:
            raise NotImplementedError("This agent only supports xDai markets")


if __name__ == "__main__":
    agent = DeployableThinkThoroughlyAgent()
    agent.deploy_local(market_type=MarketType.OMEN,
                       sleep_time=540,
                       timeout=180,
                       place_bet=False)
    # agent.deploy_gcp(
    #     repository=f"git+{get_current_git_url()}@{get_current_git_commit_sha()}",
    #     market_type=MarketType.OMEN,
    #     labels={OWNER_KEY: getpass.getuser()},
    #     secrets={
    #         "TAVILY_API_KEY": "GNOSIS_AI_TAVILY_API_KEY:latest",
    #     },
    #     memory=1024,
    #     api_keys=APIKeys(
    #         BET_FROM_ADDRESS=verify_address(
    #             "0xb611A9f02B318339049264c7a66ac3401281cc3c"
    #         ),
    #         BET_FROM_PRIVATE_KEY=private_key_type("EVAN_OMEN_BETTER_0_PKEY:latest"),
    #         OPENAI_API_KEY=SecretStr("EVAN_OPENAI_API_KEY:latest"),
    #         MANIFOLD_API_KEY=None,
    #     ),
    #     cron_schedule="0 */12 * * *",
    #     timeout=540,
    # )
