import typing as t

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent, MAX_AVAILABLE_MARKETS
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.markets.agent_market import AgentMarket, SortBy, FilterBy
from prediction_market_agent_tooling.markets.data_models import Position, ProbabilisticAnswer, Trade, TradeType, \
    BetAmount, TokenAmount
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import OmenSubgraphHandler
from prediction_market_agent_tooling.tools.langfuse_ import get_langfuse_langchain_config, observe

from prediction_market_agent.agents.arbitrage_agent.data_models import Correlation, CorrelatedMarketPair
from prediction_market_agent.db.pinecone_handler import PineconeHandler
from prediction_market_agent.utils import APIKeys


class DeployableOmenArbitrageAgent(DeployableTraderAgent):
    """ Agent that places mirror bets on Omen for (quasi) risk-neutral profit. """

    correlation_threshold: int = 0.8

    def load(self) -> None:
        self.subgraph_handler = OmenSubgraphHandler()
        self.pinecone_handler = PineconeHandler()
        self.chain = self._build_chain()

    def get_markets(
            self,
            market_type: MarketType,
            limit: int = MAX_AVAILABLE_MARKETS,
            sort_by: SortBy = SortBy.CLOSING_SOONEST,
            filter_by: FilterBy = FilterBy.OPEN,
    ) -> t.Sequence[AgentMarket]:
        return super().get_markets(market_type=market_type,
                                   limit=100,
                                   sort_by=SortBy.HIGHEST_LIQUIDITY,
                                   # Fetching most liquid markets since more likely they will have related markets
                                   filter_by=FilterBy.OPEN)

    def verify_market(self, market_type: MarketType, market: AgentMarket) -> bool:
        is_valid = super().verify_market(market_type=market_type, market=market)
        # ToDo - Additional logic - it has a correlated market.

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        # ToDo - Maybe this needs to reference correlated_markets
        return ProbabilisticAnswer(p_yes=Probability(0.5),
                                   confidence=1.0)

    def _build_chain(self) -> RunnableSerializable[t.Any, t.Any]:
        llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o-mini",
            api_key=APIKeys().openai_api_key_secretstr_v1,
        )
        prompt_template = """You are given 2 Prediction Market titles, market 1 and market 2. Your job is to output a single float number between -1 and 1, representing the correlation between the event outcomes of markets 1 and 2.
                    Correlation can be understood as the conditional probability that market 2 resolves to YES, given that market 1 resolved to YES.
                    Correlation should be a float number between -1 and 1. 

                    [MARKET 1]
                    {main_market_question}

                    [MARKET 2]
                    {related_market_question}

                    Follow the formatting instructions below for producing an output in the correct format.
                    {format_instructions}
                    """
        parser = PydanticOutputParser(pydantic_object=Correlation)
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["main_market_question", "related_market_question"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        return chain

    @observe()
    def calculate_correlation_between_markets(self, market: AgentMarket, related_market: AgentMarket) -> Correlation:
        return self.chain.invoke(
            {"main_market_question": market.question, "related_market_question": related_market.question})

    @observe()
    def is_correlation_above_threshold(self, correlation_score: float) -> bool:
        return correlation_score >= self.correlation_threshold

    @observe()
    def get_correlated_markets(self, market: AgentMarket) -> list[CorrelatedMarketPair]:

        # We try to find similar, open markets which point to the same outcome.
        correlated_markets = []
        related = self.pinecone_handler.find_nearest_questions_with_threshold(limit=10, text=market.title)
        omen_markets = self.subgraph_handler.get_omen_binary_markets(limit=len(related),
                                                                     id_in=[i.market_address.lower() for i in related],
                                                                     finalized=False,
                                                                     resolved=False)

        # Note that negative correlation is hard - e.g. for the US presidential election, markets on each candidate are not seen as -100% correlated.
        for related_market in omen_markets:
            result = self.chain.invoke({"main_market": market, "related_market": related_market},
                                       config=get_langfuse_langchain_config())
            if related_market.market_maker_contract_address_checksummed != market.market_maker_contract_address_checksummed and self.is_correlation_above_threshold(result.correlation):
                correlated_markets.append(
                    CorrelatedMarketPair(main_market=market, related_market=related_market,

                                         correlation=result.correlation))
        return correlated_markets

    @observe()
    def build_trades_for_correlated_markets(self, pair: CorrelatedMarketPair) -> list[Trade]:
        market_to_bet_yes, market_to_bet_no = pair.main_market, pair.related_market
        total_amount: BetAmount = pair.main_market.get_tiny_bet_amount()
        # Split between main_market and related_market
        amount_yes, amount_no = pair.split_bet_amount_between_yes_and_no(total_amount.amount)
        return [
            Trade(
                trade_type=TradeType.BUY, outcome=True,
                amount=TokenAmount(amount=amount_yes, currency=market_to_bet_yes.currency)),
            Trade(
                trade_type=TradeType.BUY, outcome=False,
                amount=TokenAmount(amount=amount_no, currency=market_to_bet_no.currency)),
        ]

    @observe()
    def build_trades(self, market: AgentMarket, answer: ProbabilisticAnswer, existing_position: Position | None) -> \
            list[Trade]:

        trades = []
        correlated_markets = self.get_correlated_markets(market=market)
        # For each correlated market, we want to place YES/NO or NO/YES trades.
        for pair in correlated_markets:
            trades_for_pair = self.build_trades_for_correlated_markets(pair)
            trades.extend(trades_for_pair)

        return trades
