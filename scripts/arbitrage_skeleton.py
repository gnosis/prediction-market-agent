from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from prediction_market_agent_tooling.markets.agent_market import FilterBy, SortBy
from prediction_market_agent_tooling.markets.omen.data_models import OmenMarket
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import OmenSubgraphHandler
from pydantic import BaseModel, computed_field

from prediction_market_agent.db.pinecone_handler import PineconeHandler
from prediction_market_agent.utils import APIKeys

CORRELATION_THRESHOLD = 0.8


class Correlation(BaseModel):
    correlation: float


class CorrelatedMarketPair(BaseModel):
    main_market: OmenMarket
    related_market: OmenMarket
    correlation: float

    @computed_field
    @property
    def potential_profit_per_bet_unit(self) -> float:
        """
        Calculate potential profit per bet unit based on market correlation.
        For positively correlated markets: Bet YES/NO or NO/YES.
        For negatively correlated markets: Bet YES/YES or NO/NO.
        """
        # Check correlation: +ve correlation or -ve correlation
        if self.correlation >= CORRELATION_THRESHOLD:
            # For +ve correlation, bet YES/NO or NO/YES
            p_yes = min(self.main_market.current_p_yes, self.related_market.current_p_yes)
            p_no = min(self.main_market.current_p_no, self.related_market.current_p_no)
            total_probability = p_yes + p_no
        else:
            # Smaller correlations will be handled in a future ticket
            # https://github.com/gnosis/prediction-market-agent/issues/508
            # Negative correlations are not yet supported by the current LLM prompt, hence not handling those for now.
            return 0

        # Ensure total_probability is non-zero to avoid division errors
        if total_probability > 0:
            return (1. / total_probability) - 1.
        else:
            return 0  # No arbitrage possible if the sum of probabilities is zero




if __name__ == "__main__":
    # ToDo
    # 1. Find arbitrage opportunities - find markets which are open and have same outcome (either
    # positively or negatively correlated)
    sh = OmenSubgraphHandler()
    open_markets = sh.get_omen_binary_markets_simple(limit=10, filter_by=FilterBy.OPEN,
                                                     sort_by=SortBy.HIGHEST_LIQUIDITY)
    # We now try to find similar, open markets which point to the same outcome.
    # If not the same outcome, we invert p_yes.
    pinecone_handler = PineconeHandler()
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        api_key=APIKeys().openai_api_key_secretstr_v1,
    )
    prompt_template = """You are given 2 Prediction Market titles, market 1 and market 2. Your job is to output a single float number between -1 and 1, representing the correlation between the event outcomes of markets 1 and 2.
    Correlation can be understood as the conditional probability that market 2 resolves to YES, given that market 1 resolved to YES.
    Correlation should be a float number between -1 and 1. 

    [MARKET 1]
    {main_market}

    [MARKET 2]
    {related_market}

    Follow the formatting instructions below for producing an output in the correct format.
    {format_instructions}
    """
    parser = PydanticOutputParser(pydantic_object=Correlation)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["main_market", "related_market"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    # prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = prompt | llm | parser
    correlated_markets = []
    for main_market in open_markets:
        related = pinecone_handler.find_nearest_questions_with_threshold(limit=100, text=main_market.title)
        omen_markets = sh.get_omen_binary_markets(limit=len(related),
                                                  id_in=[i.market_address.lower() for i in related], finalized=False,
                                                  resolved=False)

        # Note that negative correlation is hard - e.g. for the US presidential election, markets on each candidate
        # are not seen as -100% correlated.
        for related_market in omen_markets:
            # Todo - add langfuse config
            result = chain.invoke({"main_market": main_market, "related_market": related_market})
            print(f"{result=}")
            if related_market.market_maker_contract_address_checksummed != main_market.market_maker_contract_address_checksummed and result.correlation >= CORRELATION_THRESHOLD:
                print(
                    f'found matching markets, {main_market=}, main market prices {main_market.outcomeTokenMarginalPrices} {related_market=} related market prices {related_market.outcomeTokenMarginalPrices}')

                correlated_markets.append(CorrelatedMarketPair(main_market=main_market, related_market=related_market,
                                                               correlation=result.correlation))

    # 2. Calculate price differences - order by expected value
    correlated_markets.sort(key=lambda x: x.potential_profit_per_bet_unit, reverse=True)

    print('price diff')
    # 3. Execute trades
    print('execute trades')
