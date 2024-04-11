import random
import typing as t
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.utils import should_not_happen

from prediction_market_agent.agents.crewai_subsequential_agent.crewai_agent_subquestions import (
    CrewAIAgentSubquestions,
)
from prediction_market_agent.agents.utils import market_is_saturated


class DeployableThinkThoroughlyAgent(DeployableAgent):
    # For cheaper credits at this experimental stage
    def __init__(self) -> None:
        super().__init__()

    def pick_markets(self, markets: t.Sequence[AgentMarket]) -> t.Sequence[AgentMarket]:
        # We simply pick 5 random markets to bet on
        picked_markets: list[AgentMarket] = []
        markets = list(markets)
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
        result = CrewAIAgentSubquestions().answer_binary_market(market.question)
        return (
            True
            if result.decision == "y"
            else False
            if result.decision == "n"
            else should_not_happen()
        )


if __name__ == "__main__":
    agent = DeployableThinkThoroughlyAgent()
    agent.deploy_local(
        market_type=MarketType.OMEN, sleep_time=540, timeout=180, place_bet=False
    )
