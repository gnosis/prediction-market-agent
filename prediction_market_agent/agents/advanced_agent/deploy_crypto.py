from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.tools.google_utils import search_google_serper
from prediction_market_agent_tooling.tools.openai_utils import get_openai_provider
from prediction_market_agent_tooling.tools.utils import utcnow
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from prediction_market_agent.tools.web_scrape.markdown import web_scrape
from prediction_market_agent.utils import APIKeys


class AdvancedAgent(DeployableTraderAgent):
    """
    This is the most basic agent that should be actually able to do some evidence-based predictions.
    Use as a baseline for comparing with other agents.
    """

    bet_on_n_markets_per_run = 4





    # ADDED TO FILTER MARKETS
    def get_markets_to_trade(self, market_type):
        """
        Intercepts the market list to ensure we only bet on 
        high-liquidity Crypto and Travel markets.
        """
        # 1. Call the original method from the parent class to get all markets
        all_markets = super().get_markets_to_trade(market_type)
        
        # 2. Define our "Safe Zones" from your analysis
        target_categories = ['cryptocurrency', 'travel', 'crypto']
        crypto_keywords = ["btc", "eth", "bitcoin", "ethereum", "crypto", "cryptocurrency", "solana"]
        
        # 3. Filter the list
        filtered = [
            m for m in all_markets 
            if any(word in (m.question or "").lower() for word in crypto_keywords)
        ]
        filtered.sort(key=lambda x: x.collateral_volume, reverse=True)
        
        logger.info(f"🛡️ Thinker Filter: Reduced {len(all_markets)} markets down to {len(filtered)} safe targets.")
        return filtered






    # TODO: Uncomment and configure after we get some historic bet data
    # def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
    #     return BinaryKellyBettingStrategy(
    #         max_position_amount=get_maximum_possible_bet_amount(
    #             min_=1, max_=5, trading_balance=market.get_trade_balance(APIKeys())
    #         ),
    #         max_price_impact=0.7,
    #     )

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        # Search for results on Google
        google_results = search_google_serper(market.question)
        # Filter out Manifold results, because copy-pasting the answers isn't fun!
        # (However, generally it's allowed to use the information from other markets in your agents.)
        google_results = [url for url in google_results if "manifold" not in url]
        # If no results are found, return None, as we can't predict with nothing
        if not google_results:
            logger.info(f"No results found for {market.question}.")
            return None
        # Strip down content to fit into the context window
        contents = [
            scraped[:10000]
            for url in google_results[:5]
            if (scraped := web_scrape(url))
        ]
        # Again if no contents are scraped, return None
        if not contents:
            logger.info(f"No contents found for {market.question}")
            return None
        # And give it to the LLM to predict the probability and confidence
        probability, confidence = llm(market.question, contents)

        return ProbabilisticAnswer(
            confidence=confidence,
            p_yes=Probability(probability),
            reasoning="I asked Google and LLM to do it!",
        )


def llm(question: str, contents: list[str]) -> tuple[float, float]:
    agent = Agent(
        OpenAIModel(
            "gpt-4o-mini",
            provider=get_openai_provider(api_key=APIKeys().openai_api_key),
        ),
        system_prompt="You are a world-class crypto analyst market trading agent.",
    )
    result = agent.run_sync(
        f"""Today is {utcnow()}.

Given the following question and content from google search, what's the probability that the thing in the question will happen?

Question: {question}

Content: {contents}

Return only the probability float number and confidence float number, separated by space, nothing else."""
    ).output
    probability, confidence = map(float, result.split())
    return probability, confidence
