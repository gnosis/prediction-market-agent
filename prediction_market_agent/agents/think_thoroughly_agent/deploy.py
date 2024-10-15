from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.deploy.betting_strategy import (
    BettingStrategy,
    KellyBettingStrategy,
)
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.think_thoroughly_agent.think_thoroughly_agent import (
    ThinkThoroughlyBase,
    ThinkThoroughlyWithItsOwnResearch,
    ThinkThoroughlyWithPredictionProphetResearch,
)


class DeployableThinkThoroughlyAgentBase(DeployableTraderAgent):
    agent_class: type[ThinkThoroughlyBase]
    model: str
    bet_on_n_markets_per_run = 2

    def load(self) -> None:
        self.agent = self.agent_class(
            model=self.model, enable_langfuse=self.enable_langfuse
        )

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        return self.agent.answer_binary_market(
            market.question, created_time=market.created_time
        )

    def before_process_markets(self, market_type: MarketType) -> None:
        self.agent.pinecone_handler.update_markets()
        super().before_process_markets(market_type=market_type)


class DeployableThinkThoroughlyAgent(DeployableThinkThoroughlyAgentBase):
    agent_class = ThinkThoroughlyWithItsOwnResearch
    model: str = "gpt-4-turbo-2024-04-09"

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        return KellyBettingStrategy(max_bet_amount=5, max_price_impact=None)


class DeployableThinkThoroughlyProphetResearchAgent(DeployableThinkThoroughlyAgentBase):
    agent_class = ThinkThoroughlyWithPredictionProphetResearch
    model: str = "gpt-4-turbo-2024-04-09"

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        return KellyBettingStrategy(max_bet_amount=5, max_price_impact=0.4)


if __name__ == "__main__":
    agent = DeployableThinkThoroughlyAgent(place_bet=False)
    agent.deploy_local(
        market_type=MarketType.OMEN,
        sleep_time=540,
        timeout=180,
    )
