from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.deploy.betting_strategy import (
    BettingStrategy,
    KellyBettingStrategy,
)
from prediction_market_agent_tooling.gtypes import USD
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.think_thoroughly_agent.think_thoroughly_agent import (
    ThinkThoroughlyBase,
    ThinkThoroughlyWithItsOwnResearch,
    ThinkThoroughlyWithPredictionProphetResearch,
)
from prediction_market_agent.agents.utils import get_maximum_possible_bet_amount
from prediction_market_agent.utils import APIKeys


class DeployableThinkThoroughlyAgentBase(DeployableTraderAgent):
    agent_class: type[ThinkThoroughlyBase]
    bet_on_n_markets_per_run = 1

    def load(self) -> None:
        self.agent = self.agent_class(enable_langfuse=self.enable_langfuse)

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        return self.agent.answer_binary_market(
            market.question, created_time=market.created_time
        )

    def before_process_markets(self, market_type: MarketType) -> None:
        # self.agent.pinecone_handler.insert_all_omen_markets_if_not_exists()
        super().before_process_markets(market_type=market_type)


class DeployableThinkThoroughlyAgent(DeployableThinkThoroughlyAgentBase):
    agent_class = ThinkThoroughlyWithItsOwnResearch

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        return KellyBettingStrategy(
            max_bet_amount=get_maximum_possible_bet_amount(
                min_=USD(1),
                max_=USD(5),
                trading_balance=market.get_trade_balance(APIKeys()),
            ),
            max_price_impact=None,
        )


class DeployableThinkThoroughlyProphetResearchAgent(DeployableThinkThoroughlyAgentBase):
    agent_class = ThinkThoroughlyWithPredictionProphetResearch

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        return KellyBettingStrategy(
            max_bet_amount=get_maximum_possible_bet_amount(
                min_=USD(1),
                max_=USD(5),
                trading_balance=market.get_trade_balance(APIKeys()),
            ),
            max_price_impact=0.4,
        )


if __name__ == "__main__":
    agent = DeployableThinkThoroughlyAgent(
        place_trades=False, store_predictions=False, store_trades=False
    )
    agent.deploy_local(
        market_type=MarketType.OMEN,
        sleep_time=540,
        run_time=180,
    )
