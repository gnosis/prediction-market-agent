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
from prediction_market_agent_tooling.deploy.betting_strategy import (
    BettingStrategy,
    MaxAccuracyWithKellyScaledBetsStrategy,
)
from prediction_market_agent_tooling.gtypes import USD, Probability
from prediction_market_agent.agents.utils import get_maximum_possible_bet_amount

load_dotenv()


class MultiPersonaEnsembleAgent(DeployableTraderAgent):
    bet_on_n_markets_per_run = 5

    EDGE_THRESHOLD = 0.07 #originally 0.05
    MAX_DISAGREEMENT = 0.25

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
                "multi_persona_ensemble_agent",
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

    def run_persona(
        self,
        persona_name: str,
        persona_instruction: str,
        title: str,
        market_prob: float,
        context: str,
    ) -> dict:
        prompt = f"""
You are the {persona_name} in a prediction-market forecasting ensemble.

Your role:
{persona_instruction}

Market question:
{title}

Current market implied probability of YES:
{market_prob:.3f}

Context:
{context}

Return exactly:
Reasoning: <2-4 sentences>
Probability: [[0.XX]]
Confidence: [[0.XX]]
Skip: [[YES]] or [[NO]]
"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.choices[0].message.content or ""

        prob_match = re.search(r"Probability:\s*\[\[(\d?\.\d+)\]\]", text)
        conf_match = re.search(r"Confidence:\s*\[\[(\d?\.\d+)\]\]", text)
        skip_match = re.search(r"Skip:\s*\[\[(YES|NO)\]\]", text, re.IGNORECASE)

        p = float(prob_match.group(1)) if prob_match else 0.5
        c = float(conf_match.group(1)) if conf_match else 0.5
        skip = skip_match.group(1).upper() == "YES" if skip_match else False

        p = max(0.01, min(0.99, p))
        c = max(0.01, min(0.99, c))

        return {
            "persona": persona_name,
            "probability": p,
            "confidence": c,
            "skip": skip,
            "raw": text,
        }

    def aggregate_personas(self, persona_outputs: list[dict]) -> tuple[float, float, float, bool, str]:
        weights = {
            "Researcher": 0.35,
            "Skeptic": 0.20,
            "Trader": 0.25,
            "Risk Manager": 0.20,
        }

        weighted_prob = 0.0
        weighted_conf = 0.0
        total_weight = 0.0
        probs = []
        skip_votes = 0
        reasoning_parts = []

        for output in persona_outputs:
            name = output["persona"]
            w = weights.get(name, 0.25)

            weighted_prob += w * output["probability"]
            weighted_conf += w * output["confidence"]
            total_weight += w
            probs.append(output["probability"])
            reasoning_parts.append(f"{name}: {output['raw']}")

            if output["skip"]:
                skip_votes += 1
            print(f"{name}: p={output['probability']:.3f}, conf={output['confidence']:.3f}, skip={output['skip']}")

        final_prob = weighted_prob / total_weight
        final_conf = weighted_conf / total_weight
        disagreement = max(probs) - min(probs) if probs else 0.0
        should_skip = skip_votes >= 2

        combined_reasoning = "\n\n".join(reasoning_parts)

        return final_prob, final_conf, disagreement, should_skip, combined_reasoning

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        title = market.question
        market_prob = float(market.p_yes)
        context = self.get_live_context(title)

        personas = [
            (
                "Researcher",
                "Focus on current evidence and news. Estimate what the facts suggest."
            ),
            (
                "Skeptic",
                "Challenge the obvious conclusion. Look for missing assumptions, bad evidence, and reasons the event may not happen."
            ),
            (
                "Trader",
                "Think like a market participant. Compare likely reality to current market pricing and identify whether the market seems over- or under-priced."
            ),
            (
                "Risk Manager",
                "Focus on uncertainty, ambiguity, unreliable evidence, and whether this market should be skipped."
            ),
        ]

        persona_outputs = []
        for name, instruction in personas:
            output = self.run_persona(
                persona_name=name,
                persona_instruction=instruction,
                title=title,
                market_prob=market_prob,
                context=context,
            )
            persona_outputs.append(output)

        final_prob, final_conf, disagreement, should_skip, reasoning = self.aggregate_personas(persona_outputs)

        edge = abs(final_prob - market_prob)

        print(f"\nAnalyzing: {title}")
        print(f"Market p_yes:   {market_prob:.3f}")
        print(f"Final p_yes:    {final_prob:.3f}")
        print(f"Confidence:     {final_conf:.3f}")
        print(f"Disagreement:   {disagreement:.3f}")
        print(f"Edge:           {edge:.3f}")
        print(f"Risk skip vote: {should_skip}")

        
        
        # if should_skip:
        #     print("Skipping: risk-manager veto / skip vote.")
        #     return None

        # if disagreement > self.MAX_DISAGREEMENT:
        #     print("Skipping: personas disagree too much.")
        #     return None

        # if edge < self.EDGE_THRESHOLD:
        #     print("Skipping: edge too small.")
        #     return None

        # return ProbabilisticAnswer(
        #     p_yes=Probability(final_prob),
        #     confidence=final_conf,
        #     reasoning=reasoning,
        # )

        trade_side = "YES" if final_prob > market_prob else "NO"

        if should_skip:
            print("Skipping: risk-manager veto / skip vote.")
            self.log_forecast(
                market=market,
                market_prob=market_prob,
                final_prob=final_prob,
                confidence=final_conf,
                disagreement=disagreement,
                traded=False,
                trade_side="SKIP_RISK",
            )
            return None

        if disagreement > self.MAX_DISAGREEMENT:
            print("Skipping: personas disagree too much.")
            self.log_forecast(
                market=market,
                market_prob=market_prob,
                final_prob=final_prob,
                confidence=final_conf,
                disagreement=disagreement,
                traded=False,
                trade_side="SKIP_DISAGREEMENT",
            )
            return None

        if edge < self.EDGE_THRESHOLD:
            print("Skipping: edge too small.")
            self.log_forecast(
                market=market,
                market_prob=market_prob,
                final_prob=final_prob,
                confidence=final_conf,
                disagreement=disagreement,
                traded=False,
                trade_side="SKIP_EDGE",
            )
            return None

        self.log_forecast(
            market=market,
            market_prob=market_prob,
            final_prob=final_prob,
            confidence=final_conf,
            disagreement=disagreement,
            traded=True,
            trade_side=trade_side,
        )

        return ProbabilisticAnswer(
            p_yes=Probability(final_prob),
            confidence=final_conf,
            reasoning=reasoning,
        )

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        return MaxAccuracyWithKellyScaledBetsStrategy(
            max_position_amount=get_maximum_possible_bet_amount(
                min_=USD(0.02),
                max_=USD(0.07),
                trading_balance=market.get_trade_balance(self.api_keys),
            ),
        )