import typing as t

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from prediction_market_agent_tooling.deploy.agent import (
    MAX_AVAILABLE_MARKETS,
    DeployableTraderAgent,
)
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import (
    AgentMarket,
    FilterBy,
    SortBy,
)
from prediction_market_agent_tooling.markets.data_models import (
    BetAmount,
    Position,
    ProbabilisticAnswer,
    TokenAmount,
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
    total_trade_amount = BetAmount(amount=0.1, currency=OmenAgentMarket.currency)
    bet_on_n_markets_per_run = 5
    max_related_markets_per_market = 10

    def run(self, market_type: MarketType) -> None:
        if market_type != MarketType.OMEN:
            raise RuntimeError(
                "Can arbitrage only on Omen since related markets embeddings available only for Omen markets."
            )
        self.subgraph_handler = OmenSubgraphHandler()
        self.pinecone_handler = PineconeHandler()
        self.pinecone_handler.update_markets()
        self.chain = self._build_chain()
        super().run(market_type=market_type)

    def get_markets(
        self,
        market_type: MarketType,
        limit: int = MAX_AVAILABLE_MARKETS,
        sort_by: SortBy = SortBy.CLOSING_SOONEST,
        filter_by: FilterBy = FilterBy.OPEN,
    ) -> t.Sequence[AgentMarket]:
        return super().get_markets(
            market_type=market_type,
            limit=50,
            sort_by=SortBy.HIGHEST_LIQUIDITY,
            # Fetching most liquid markets since more likely they will have related markets
            filter_by=FilterBy.OPEN,
        )

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        return ProbabilisticAnswer(p_yes=Probability(0.5), confidence=1.0)

    def _build_chain(self) -> RunnableSerializable[t.Any, t.Any]:
        llm = ChatOpenAI(
            temperature=0,
            model=self.model,
            api_key=APIKeys().openai_api_key_secretstr_v1,
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
            limit=self.max_related_markets_per_market * 10, text=market.question
        )

        omen_markets = self.subgraph_handler.get_omen_binary_markets(
            limit=len(related),
            id_in=[i.market_address for i in related],
            resolved=False,
        )

        # Order omen_markets in the same order as related
        related_market_addresses = [i.market_address for i in related]
        omen_markets = sorted(
            omen_markets, key=lambda m: related_market_addresses.index(m.id)
        )
        omen_markets = [m for m in omen_markets if m.id != market.id]
        print(f"Fetched {len(omen_markets)} related markets for market {market.id}")

        for related_market in omen_markets:
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
        market_to_bet_yes, market_to_bet_no = pair.main_market, pair.related_market

        # Split between main_market and related_market
        arbitrage_bet = pair.split_bet_amount_between_yes_and_no(
            self.total_trade_amount.amount
        )

        main_trade = Trade(
            trade_type=TradeType.BUY,
            outcome=arbitrage_bet.main_market_bet.direction,
            amount=TokenAmount(
                amount=arbitrage_bet.main_market_bet.size,
                currency=pair.main_market.currency,
            ),
        )

        # related trade
        related_trade = Trade(
            trade_type=TradeType.BUY,
            outcome=arbitrage_bet.related_market_bet.direction,
            amount=TokenAmount(
                amount=arbitrage_bet.related_market_bet.size,
                currency=pair.related_market.currency,
            ),
        )

        trades = [main_trade, related_trade]
        logger.info(f"Placing arbitrage trades {trades}")
        return trades

    @observe()
    def build_trades(
        self,
        market: AgentMarket,
        answer: ProbabilisticAnswer,
        existing_position: Position | None,
    ) -> list[Trade]:
        trades = []
        correlated_markets = self.get_correlated_markets(market=market)
        for pair in correlated_markets:
            if pair.potential_profit_per_bet_unit > 0:
                trades_for_pair = self.build_trades_for_correlated_markets(pair)
                trades.extend(trades_for_pair)

        return trades
