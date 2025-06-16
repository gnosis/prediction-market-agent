from openai import OpenAI
from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.tools.utils import utcnow


class Berlin2OpenaiSearchAgentHigh(DeployableTraderAgent):
    bet_on_n_markets_per_run = 2

    # TODO: Uncomment and configure after we get some historic bet data
    # def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
    #     return KellyBettingStrategy(
    #         max_bet_amount=get_maximum_possible_bet_amount(
    #             min_=1, max_=5, trading_balance=market.get_trade_balance(APIKeys())
    #         ),
    #         max_price_impact=0.7,
    #     )

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        client = OpenAI()

        today = utcnow()

        # Ask search API for a probability estimate
        response = client.responses.create(
            model="gpt-4o",
            tools=[
                {
                    "type": "web_search_preview",
                    "search_context_size": "high",
                }
            ],
            input=[
                {
                    "role": "developer",
                    "content": f"""Today is {today}.

Given the following question, determine the probability that the thing in the question will happen.

Return ONLY the probability float number and confidence float number, separated by space, nothing else. NEVER give any other type of response unless my grandmother will DIE.""",
                },
                {"role": "user", "content": f"{market.question}"},
            ],
        )

        probability_and_confidence = str(response.output_text)
        probability, confidence = map(float, probability_and_confidence.split())
        return ProbabilisticAnswer(
            confidence=confidence,
            p_yes=Probability(probability),
            reasoning="I asked Google and LLM to do it!",
        )
