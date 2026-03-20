import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient
import csv
from datetime import datetime, timezone
from pathlib import Path
from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.gtypes import Probability

load_dotenv()


class MultiPersonaAgent(DeployableTraderAgent):
    bet_on_n_markets_per_run = 3
    EDGE_THRESHOLD = 0.05

    def log_forecast(
        self,
        market: AgentMarket,
        market_prob: float,
        final_prob: float,
        confidence: float,
        disagreement: float,
        traded: bool,
        trade_side: str,
    ) -> None:
        log_path = Path("forecast_log.csv")
        file_exists = log_path.exists()

        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "timestamp_utc",
                    "agent_name",
                    "market_id",
                    "question",
                    "market_prob",
                    "final_prob",
                    "confidence",
                    "disagreement",
                    "traded",
                    "trade_side",
                    "outcome",
                ])

            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                "multi_persona_agent",
                str(market.id),
                market.question,
                round(market_prob, 6),
                round(final_prob, 6),
                round(confidence, 6),
                round(disagreement, 6),
                int(traded),
                trade_side,
                "",  # fill in later when market resolves
            ])

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

        #ADDED  
        edge = abs(p_yes - current_price)
        disagreement = 0.0
        confidence = 0.7
        trade_side = "YES" if p_yes > current_price else "NO"

        ###

        print(f"\nAnalyzing: {title}")
        print(f"Market p_yes: {current_price:.3f}")
        print(f"Model p_yes:  {p_yes:.3f}")
        print(f"Edge:         {p_yes - current_price:.3f}")

        # return ProbabilisticAnswer(
        #     p_yes=Probability(p_yes),
        #     confidence=0.7,
        #     reasoning=text,
        # )


        #ADDED
        if edge < self.EDGE_THRESHOLD:
            print("Skipping: edge too small.")
            self.log_forecast(
                market=market,
                market_prob=current_price,
                final_prob=p_yes,
                confidence=confidence,
                disagreement=disagreement,
                traded=False,
                trade_side="SKIP_EDGE",
            )
            return None

        self.log_forecast(
            market=market,
            market_prob=current_price,
            final_prob=p_yes,
            confidence=confidence,
            disagreement=disagreement,
            traded=True,
            trade_side=trade_side,
        )

        return ProbabilisticAnswer(
            p_yes=Probability(p_yes),
            confidence=confidence,
            reasoning=text,
        )
