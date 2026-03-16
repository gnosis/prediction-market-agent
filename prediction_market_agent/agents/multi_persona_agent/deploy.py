import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.gtypes import Probability

load_dotenv()


class MultiPersonaEnsembleAgent(DeployableTraderAgent):
    bet_on_n_markets_per_run = 2

    def load(self) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def get_live_context(self, query: str) -> str:
        result = self.tavily.search(
            query=query,
            search_depth="advanced",
            max_results=5,
        )
        return "\n".join(
            item.get("content", "") for item in result.get("results", [])
        )

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        title = market.question
        current_price = float(market.p_yes)
        context = self.get_live_context(title)

        prompt = f"""
You are a multi-persona ensemble prediction agent.

Market question:
{title}

Current market implied probability:
{current_price:.3f}

Context:
{context}

Simulate these personas:
1. Researcher: what do current sources suggest?
2. Skeptic: why might the obvious interpretation be wrong?
3. Trader: is the market already pricing this in?
4. Risk manager: should this market be skipped?

Return:
- brief reasoning
- final probability in exactly this format: [[0.XX]]
"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.choices[0].message.content or ""
        match = re.search(r"\[\[(\d?\.\d+)\]\]", text)
        p_yes = float(match.group(1)) if match else 0.5
        p_yes = max(0.01, min(0.99, p_yes))

        print(f"\nAnalyzing: {title}")
        print(f"Market p_yes: {current_price:.3f}")
        print(f"Model p_yes:  {p_yes:.3f}")
        print(f"Edge:         {p_yes - current_price:.3f}")

        return ProbabilisticAnswer(
            p_yes=Probability(p_yes),
            confidence=0.7,
            reasoning=text,
        )