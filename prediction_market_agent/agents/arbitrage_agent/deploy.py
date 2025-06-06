import typing as t
from datetime import timedelta

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.gtypes import USD, Probability
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import (
    CategoricalProbabilisticAnswer,
    Position,
    ProbabilisticAnswer,
    Trade,
    TradeType,
)
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from prediction_market_agent_tooling.tools.langfuse_ import (
    get_langfuse_langchain_config,
    observe,
)
from prediction_market_agent_tooling.tools.utils import utcnow

from prediction_market_agent.agents.arbitrage_agent.data_models import (
    CorrelatedMarketPair,
    Correlation,
)
from prediction_market_agent.agents.arbitrage_agent.prompt import PROMPT_TEMPLATE
from prediction_market_agent.db.pinecone_handler import PineconeHandler
from prediction_market_agent.utils import APIKeys


class DeployableArbitrageAgent(DeployableTraderAgent):
    """Agent that places mirror bets on Omen for (quasi) risk-neutral profit."""

    model = "gpt-4o"
    # trade amount will be divided between correlated markets.
    total_trade_amount = USD(0.1)
    bet_on_n_markets_per_run = 5
    max_related_markets_per_market = 10
    n_markets_to_fetch = 50

    def run(self, market_type: MarketType) -> None:
        if market_type != MarketType.OMEN:
            raise RuntimeError(
                "Can arbitrage only on Omen since related markets embeddings available only for Omen markets."
            )
        self.subgraph_handler = OmenSubgraphHandler()
        self.pinecone_handler = PineconeHandler()
        self.pinecone_handler.insert_all_omen_markets_if_not_exists()
        self.chain = self._build_chain()
        super().run(market_type=market_type)

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        return ProbabilisticAnswer(p_yes=Probability(0.5), confidence=1.0)

    def _build_chain(self) -> RunnableSerializable[t.Any, t.Any]:
        llm = ChatOpenAI(
            temperature=0,
            model_name=self.model,
            openai_api_key=APIKeys().openai_api_key,
        )

        parser = PydanticOutputParser(pydantic_object=Correlation)
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["main_market_question", "related_market_question"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        return chain

    @observe()
    def calculate_correlation_between_markets(
        self, market: AgentMarket, related_market: AgentMarket
    ) -> Correlation:
        correlation: Correlation = self.chain.invoke(
            {
                "main_market_question": market.question,
                "related_market_question": related_market.question,
            }
        )
        return correlation

    @observe()
    def get_correlated_markets(self, market: AgentMarket) -> list[CorrelatedMarketPair]:
        # We try to find similar, open markets which point to the same outcome.
        correlated_markets = []
        # We only wanted to find related markets that are open.
        # We intentionally query more markets in the hope it yields open markets.
        # We could store market_status (open, closed) in Pinecone, but we refrain from it
        # to keep the chain data (or graph) as the source-of-truth, instead of managing the
        # update process of the vectorDB.

        related = self.pinecone_handler.find_nearest_questions_with_threshold(
            limit=self.max_related_markets_per_market,
            text=market.question,
            filter_on_metadata={
                "close_time_timestamp": {
                    "$gte": int((utcnow() + timedelta(hours=1)).timestamp())
                }
            },
        )

        omen_markets = self.subgraph_handler.get_omen_markets(
            limit=len(related),
            id_in=[i.market_address for i in related if i.market_address != market.id],
            resolved=False,
        )

        # Order omen_markets in the same order as related
        related_market_addresses = [i.market_address for i in related]
        omen_markets = sorted(
            omen_markets, key=lambda m: related_market_addresses.index(m.id)
        )

        logger.info(
            f"Fetched {len(omen_markets)} related markets for market {market.id}"
        )

        for related_market in omen_markets:
            if related_market.id.lower() == market.id.lower():
                logger.info(
                    f"Skipping related market {related_market.id} since same market as {market.id}"
                )
                continue
            result: Correlation = self.chain.invoke(
                {
                    "main_market_question": market,
                    "related_market_question": related_market,
                },
                config=get_langfuse_langchain_config(),
            )
            if result.near_perfect_correlation is not None:
                related_agent_market = OmenAgentMarket.from_data_model(related_market)
                correlated_markets.append(
                    CorrelatedMarketPair(
                        main_market=market,
                        related_market=related_agent_market,
                        correlation=result,
                    )
                )
        return correlated_markets

    @observe()
    def build_trades_for_correlated_markets(
        self, pair: CorrelatedMarketPair
    ) -> list[Trade]:
        # Split between main_market and related_market
        arbitrage_bet = pair.split_bet_amount_between_yes_and_no(
            self.total_trade_amount
        )

        main_trade = Trade(
            trade_type=TradeType.BUY,
            outcome=arbitrage_bet.main_market_bet.direction,
            amount=arbitrage_bet.main_market_bet.size,
        )

        # related trade
        related_trade = Trade(
            trade_type=TradeType.BUY,
            outcome=arbitrage_bet.related_market_bet.direction,
            amount=arbitrage_bet.related_market_bet.size,
        )

        trades = [main_trade, related_trade]
        logger.info(f"Placing arbitrage trades {trades}")
        return trades

    @observe()
    def build_trades(
        self,
        market: AgentMarket,
        answer: CategoricalProbabilisticAnswer,
        existing_position: Position | None,
    ) -> list[Trade]:
        trades = []
        correlated_markets = self.get_correlated_markets(market=market)
        for pair in correlated_markets:
            if pair.main_market_and_related_market_equal:
                logger.info(
                    "Skipping market pair since related- and main market are the same."
                )
                continue
            # We want to profit at least 0.5% per market (value chosen as initial baseline).
            if pair.potential_profit_per_bet_unit() > 0.005:
                trades_for_pair = self.build_trades_for_correlated_markets(pair)
                trades.extend(trades_for_pair)

        return trades
