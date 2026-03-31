import os
import re
import csv
import pandas as pd
import joblib
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient
from datetime import datetime, timezone
from pathlib import Path

from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.gtypes import Probability, USD
from prediction_market_agent_tooling.deploy.betting_strategy import (
    BettingStrategy,
    MaxAccuracyWithKellyScaledBetsStrategy,
)
from prediction_market_agent.agents.utils import get_maximum_possible_bet_amount

load_dotenv()


""" 
p_market = raw market implied probability
p_cal = calibrated market probability
p_ml = ML predicted probability
p_base = blend of p_cal and p_ml
delta_llm = small bounded LLM adjustment
p_final = p_base + delta_llm

Then, edge = p_final - p_market
"""




class FusionAgent(DeployableTraderAgent):
    bet_on_n_markets_per_run = 4

    EDGE_THRESHOLD = 0.05
    MIN_CONFIDENCE = 0.50
    MAX_LLM_ADJUSTMENT = 0.05

    def load(self) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

        # Load your trained ML model here.
        # This should be something like logistic regression, random forest, etc.
        # trained on the exact same features returned by build_features().
        self.ml_model = joblib.load("ml_model.joblib")

        self.calibration_model = joblib.load("market_calibrator.joblib")

        # Optional:
        # If you later train a calibration model such as isotonic regression,
        # uncomment this and use it inside calibrate_market_prob().
        # self.calibration_model = joblib.load("market_calibrator.joblib")

    def log_forecast(
        self,
        market: AgentMarket,
        market_prob: float,
        calibrated_prob: float,
        ml_prob: float,
        llm_adjustment: float,
        final_prob: float,
        confidence: float,
        traded: bool,
        trade_side: str,
        skip_reason: str = "",
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
                    "calibrated_prob",
                    "ml_prob",
                    "llm_adjustment",
                    "final_prob",
                    "confidence",
                    "traded",
                    "trade_side",
                    "skip_reason",
                    "outcome",
                ])

            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                "hybrid_agent",
                str(market.id),
                market.question,
                round(market_prob, 6),
                round(calibrated_prob, 6),
                round(ml_prob, 6),
                round(llm_adjustment, 6),
                round(final_prob, 6),
                round(confidence, 6),
                int(traded),
                trade_side,
                skip_reason,
                "",  # fill later once market resolves
            ])

    def get_live_context(self, query: str) -> str:
        result = self.tavily.search(
            query=query,
            search_depth="advanced",
            max_results=5,
        )
        return "\n".join(
            item.get("content", "") for item in result.get("results", [])
        )

    def calibrate_market_prob(self, p_market: float) -> float:
        """
        Temporary hand-built calibration.
        Replace later with a learned calibration model if you train one.

        This reflects your finding that high YES probabilities may be somewhat inflated.
        """
        # if p_market >= 0.85:
        #     p_cal = p_market - 0.08
        # elif p_market >= 0.70:
        #     p_cal = p_market - 0.05
        # elif p_market >= 0.55:
        #     p_cal = p_market - 0.03
        # else:
        #     p_cal = p_market

        # return max(0.01, min(0.99, p_cal))

        p_cal = float(self.calibration_model.predict([p_market])[0])
        return max(0.01, min(0.99, p_cal))

        # If using a real calibration model later, do this instead:
        # p_cal = float(self.calibration_model.predict([[p_market]])[0])
        # return max(0.01, min(0.99, p_cal))

    def build_features(self, market, p_market, p_cal) -> dict:
        volume = float(getattr(market, "volume", 0.0) or 0.0)

        created_time = getattr(market, "created_time", None) or getattr(market, "creation_datetime", None)
        close_time = getattr(market, "close_time", None)

        duration_hours = 0.0
        if created_time is not None and close_time is not None:
            if created_time.tzinfo is None:
                created_time = created_time.replace(tzinfo=timezone.utc)
            if close_time.tzinfo is None:
                close_time = close_time.replace(tzinfo=timezone.utc)
            duration_hours = max((close_time - created_time).total_seconds() / 3600, 0.0)

        category = getattr(market, "category", None) or "unknown"

        return {
            "prob_at_close": p_market,
            "calibrated_prob": p_cal,
            "volume": volume,
            "duration_hours": duration_hours,
            "category": category,
        }

    def get_ml_baseline(self, market, p_market, p_cal) -> float:
        feature_dict = self.build_features(market, p_market, p_cal)
        X = pd.DataFrame([feature_dict])
        p_ml = float(self.ml_model.predict_proba(X)[0][1])
        return max(0.01, min(0.99, p_ml))

    def get_llm_overlay(
        self,
        title: str,
        market_prob: float,
        calibrated_prob: float,
        ml_prob: float,
        context: str,
    ) -> dict:
        prompt = f"""
You are helping a prediction-market trading agent.

Your task is NOT to produce a brand-new standalone forecast.
Instead, review the live context and suggest a SMALL adjustment
to the baseline forecast.

Market question:
{title}

Raw market implied probability of YES:
{market_prob:.3f}

Calibrated market probability of YES:
{calibrated_prob:.3f}

ML baseline probability of YES:
{ml_prob:.3f}

Live context:
{context}

Instructions:
- Suggest an adjustment to the baseline in the range [-0.08, +0.08]
- Positive adjustment means increase belief in YES
- Negative adjustment means decrease belief in YES
- Use small adjustments unless the evidence is unusually strong
- If evidence is stale, noisy, weak, or ambiguous, recommend skip

Return exactly in this format:
Reasoning: <2-4 sentences>
Adjustment: [[-0.03]]
Confidence: [[0.72]]
Skip: [[YES]] or [[NO]]
"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.choices[0].message.content or ""

        adj_match = re.search(r"Adjustment:\s*\[\[([+-]?\d?\.\d+)\]\]", text)
        conf_match = re.search(r"Confidence:\s*\[\[(\d?\.\d+)\]\]", text)
        skip_match = re.search(r"Skip:\s*\[\[(YES|NO)\]\]", text, re.IGNORECASE)

        adjustment = float(adj_match.group(1)) if adj_match else 0.0
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        skip = skip_match.group(1).upper() == "YES" if skip_match else False

        adjustment = max(-self.MAX_LLM_ADJUSTMENT, min(self.MAX_LLM_ADJUSTMENT, adjustment))
        confidence = max(0.01, min(0.99, confidence))

        return {
            "adjustment": adjustment,
            "confidence": confidence,
            "skip": skip,
            "raw": text,
        }

    def combine_predictions(
        self,
        p_cal: float,
        p_ml: float,
        llm_adjustment: float,
        llm_confidence: float,
    ) -> tuple[float, float]:
        """
        Stable backbone = calibrated market + ML baseline
        LLM only nudges the result.
        """
        p_base = 0.5 * p_cal + 0.5 * p_ml

        # Confidence-scaled LLM overlay
        scaled_adjustment = llm_adjustment * llm_confidence

        p_final = p_base + scaled_adjustment
        p_final = max(0.01, min(0.99, p_final))

        return p_final, scaled_adjustment

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        title = market.question
        market_prob = float(market.p_yes)

        # 1. Calibrate raw market probability
        calibrated_prob = self.calibrate_market_prob(market_prob)

        # 2. ML baseline from structured features
        ml_prob = self.get_ml_baseline(
            market=market,
            p_market=market_prob,
            p_cal=calibrated_prob,
        )

        # 3. Pull live context
        context = self.get_live_context(title)

        # 4. LLM overlay
        llm_result = self.get_llm_overlay(
            title=title,
            market_prob=market_prob,
            calibrated_prob=calibrated_prob,
            ml_prob=ml_prob,
            context=context,
        )

        # 5. Combine all components
        final_prob, scaled_adjustment = self.combine_predictions(
            p_cal=calibrated_prob,
            p_ml=ml_prob,
            llm_adjustment=llm_result["adjustment"],
            llm_confidence=llm_result["confidence"],
        )

        final_conf = llm_result["confidence"]
        reasoning = llm_result["raw"]

        edge = abs(final_prob - market_prob)
        trade_side = "YES" if final_prob > market_prob else "NO"

        print(f"\nAnalyzing: {title}")
        print(f"Market p_yes:      {market_prob:.3f}")
        print(f"Calibrated p_yes:  {calibrated_prob:.3f}")
        print(f"ML baseline p_yes: {ml_prob:.3f}")
        print(f"LLM adj scaled:    {scaled_adjustment:.3f}")
        print(f"Final p_yes:       {final_prob:.3f}")
        print(f"Confidence:        {final_conf:.3f}")
        print(f"Edge:              {edge:.3f}")
        print(f"LLM skip:          {llm_result['skip']}")

        if llm_result["skip"]:
            print("Skipping: LLM flagged weak or ambiguous evidence.")
            self.log_forecast(
                market=market,
                market_prob=market_prob,
                calibrated_prob=calibrated_prob,
                ml_prob=ml_prob,
                llm_adjustment=scaled_adjustment,
                final_prob=final_prob,
                confidence=final_conf,
                traded=False,
                trade_side="SKIP",
                skip_reason="LLM_SKIP",
            )
            return None

        if final_conf < self.MIN_CONFIDENCE:
            print("Skipping: confidence too low.")
            self.log_forecast(
                market=market,
                market_prob=market_prob,
                calibrated_prob=calibrated_prob,
                ml_prob=ml_prob,
                llm_adjustment=scaled_adjustment,
                final_prob=final_prob,
                confidence=final_conf,
                traded=False,
                trade_side="SKIP",
                skip_reason="LOW_CONFIDENCE",
            )
            return None

        if edge < self.EDGE_THRESHOLD:
            print("Skipping: edge too small.")
            self.log_forecast(
                market=market,
                market_prob=market_prob,
                calibrated_prob=calibrated_prob,
                ml_prob=ml_prob,
                llm_adjustment=scaled_adjustment,
                final_prob=final_prob,
                confidence=final_conf,
                traded=False,
                trade_side="SKIP",
                skip_reason="EDGE_TOO_SMALL",
            )
            return None

        self.log_forecast(
            market=market,
            market_prob=market_prob,
            calibrated_prob=calibrated_prob,
            ml_prob=ml_prob,
            llm_adjustment=scaled_adjustment,
            final_prob=final_prob,
            confidence=final_conf,
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
                min_=USD(0.01),
                max_=USD(0.04),
                trading_balance=market.get_trade_balance(self.api_keys),
            ),
        )

"""
A few important things you’ll still need to do:

Train and save ml_model.joblib
even a simple logistic regression is fine for version 1
Make sure build_features() matches exactly how the model was trained
Update the agent registration / run command so this new class is the one being called
Your current calibration is just a placeholder, so later you’ll probably want isotonic regression or a learned calibration map

The easiest way to test this before your real ML model is ready is to temporarily replace get_ml_baseline() with:

def get_ml_baseline(self, market, p_market, p_cal):
    return p_cal

"""