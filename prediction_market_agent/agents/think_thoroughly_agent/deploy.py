import typing as t
from datetime import timedelta

from loguru import logger
from prediction_market_agent_tooling.config import PrivateCredentials
from prediction_market_agent_tooling.deploy.agent import Answer, DeployableAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.manifold.api import (
    get_authenticated_user,
    get_manifold_bets,
    get_manifold_market,
)
from prediction_market_agent_tooling.markets.manifold.manifold import (
    ManifoldAgentMarket,
)
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from prediction_market_agent_tooling.tools.is_predictable import is_predictable_binary
from prediction_market_agent_tooling.tools.utils import should_not_happen, utcnow

from prediction_market_agent.agents.think_thoroughly_agent.think_thoroughly_agent import (
    CrewAIAgentSubquestions,
)
from prediction_market_agent.utils import APIKeys


class DeployableThinkThoroughlyAgent(DeployableAgent):
    model: str = "gpt-4-turbo-2024-04-09"
    max_markets_per_run = 5

    def load(self) -> None:
        self.agent = CrewAIAgentSubquestions(model=self.model)

    def recently_betted(self, market: AgentMarket) -> bool:
        start_time = utcnow() - timedelta(hours=24)
        keys = APIKeys()
        credentials = PrivateCredentials.from_api_keys(keys)
        recently_betted_questions = (
            set(
                get_manifold_market(b.contractId).question
                for b in get_manifold_bets(
                    user_id=get_authenticated_user(
                        keys.manifold_api_key.get_secret_value()
                    ).id,
                    start_time=start_time,
                    end_time=None,
                )
            )
            if isinstance(market, ManifoldAgentMarket)
            else (
                set(
                    b.title
                    for b in OmenSubgraphHandler().get_bets(
                        better_address=credentials.public_key,
                        start_time=start_time,
                    )
                )
                if isinstance(market, OmenAgentMarket)
                else should_not_happen(f"Uknown market: {market}")
            )
        )
        return market.question in recently_betted_questions

    def pick_markets(self, markets: t.Sequence[AgentMarket]) -> t.Sequence[AgentMarket]:
        picked_markets: list[AgentMarket] = []
        for market in markets:
            logger.info(f"Looking if we recently bet on '{market.question}'.")
            if self.recently_betted(market):
                logger.info("Recently betted, skipping.")
                continue
            logger.info(f"Verifying market predictability for '{market.question}'.")
            if is_predictable_binary(market.question):
                logger.info(f"Market '{market.question}' is predictable.")
                picked_markets.append(market)
            if len(picked_markets) >= self.max_markets_per_run:
                break
        return picked_markets

    def answer_binary_market(self, market: AgentMarket) -> Answer | None:
        return self.agent.answer_binary_market(market.question)


if __name__ == "__main__":
    agent = DeployableThinkThoroughlyAgent()
    agent.deploy_local(
        market_type=MarketType.OMEN, sleep_time=540, timeout=180, place_bet=False
    )
