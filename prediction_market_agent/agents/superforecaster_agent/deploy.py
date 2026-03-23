from openai import OpenAI
from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.deploy.betting_strategy import (
    BettingStrategy,
    MaxAccuracyWithKellyScaledBetsStrategy,
)
from prediction_market_agent_tooling.gtypes import USD, Probability
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.tools.utils import utcnow
from pydantic_ai.exceptions import UnexpectedModelBehavior

from prediction_market_agent.agents.utils import get_maximum_possible_bet_amount


# ---------------------------------------------------------------------------
# Stage 1 prompt — evidence gathering, superforecaster-aware
# ---------------------------------------------------------------------------
# The search model's job is purely retrieval and organization. We guide it to
# surface the specific *types* of evidence that matter for calibrated forecasting
# (base rates, historical precedents, reference class data) without asking it
# to draw conclusions. That judgment is reserved entirely for stage 2.

SEARCH_DEVELOPER_PROMPT = """
Today is {today}.

You will be given a prediction market question. Your task is to gather and
organize every piece of evidence that a skilled forecaster would need to
estimate the probability of the described event.

Structure your report under these headings:

BASE RATES & HISTORICAL PRECEDENTS
- How often have similar events occurred historically?
- Find concrete numbers where possible (e.g. "In the last 10 elections, the
  incumbent won 7 times" or "FDA approval rate for Phase 3 oncology trials is ~50%").
- Identify the most appropriate reference class for this question.

CURRENT SITUATION
- What is the current state of affairs directly relevant to this question?
- Key recent developments, announcements, or data points.

FACTORS FAVOURING YES
- Specific evidence or conditions that make the event more likely.

FACTORS FAVOURING NO
- Specific evidence or conditions that make the event less likely.

UNCERTAINTY & INFORMATION GAPS
- What is unknown or contested?
- Are there upcoming events that could change the picture?

Rules:
- Do not provide links — explain all evidence in full.
- Do not draw a conclusion or state a probability. That is done elsewhere.
- Include all relevant evidence even if it seems to point in different directions.
- Be precise: prefer numbers, dates, and named sources over vague qualitative claims.
""".strip()


# ---------------------------------------------------------------------------
# Stage 2 prompt — superforecaster reasoning with o3-mini
# ---------------------------------------------------------------------------
# o3-mini with high reasoning effort handles multi-step structured reasoning
# well. We give it the full Tetlock framework and ask it to work through each
# step explicitly before committing to a number. The output format stays
# minimal (float float) to match the original agent's parser.

REASONING_DEVELOPER_PROMPT = """
Today is {today}.

You are a Superforecaster trained in Philip Tetlock's Good Judgment Project
methods. You will be given a prediction market question and a structured
evidence report. Your task is to produce a well-calibrated probability
estimate.

Work through these steps before giving your answer:

1. OUTSIDE VIEW
   - What base rate or historical frequency does the evidence report provide?
   - State this as your starting probability anchor.

2. REFERENCE CLASS CHECK
   - Is this the most appropriate reference class, or should you adjust to a
     narrower or broader one? Briefly justify.

3. INSIDE VIEW ADJUSTMENTS
   - Which specific factors in the evidence push the probability UP from the
     base rate, and by roughly how much?
   - Which factors push it DOWN, and by roughly how much?
   - Net the adjustments into a revised probability.

4. SYNTHESIS
   - Combine the outside and inside views. Weight them: outside view should
     anchor you unless inside-view evidence is strong and specific.

5. BIAS CHECK
   - Are you anchoring too heavily on a vivid recent event? (recency bias)
   - Are you overweighting a memorable but unrepresentative example? (availability bias)
   - Are you being pulled toward the market's implied probability? (anchoring)
   - Correct for any bias you identify before finalising.
   - Do not discount inside-view evidence just because the scenario seems dramatic — if the evidence is specific and credible, weight it accordingly

6. CONFIDENCE
   - How much does your answer depend on uncertain or incomplete information?
   - Express this as a confidence float: 1.0 = near certainty, 0.0 = pure guess.
   - Be conservative: most real-world forecasts deserve confidence < 0.85.

Calibration rules you must follow:
- Avoid probabilities below 0.05 or above 0.95 unless the evidence is
  overwhelming and unambiguous. Justify any value outside 0.1–0.9.
- Do not let the current market price substitute for your own reasoning.
- Uncertainty is not a reason to default to 0.5 — use your base rate instead.

Return ONLY two floats separated by a single space: probability confidence
Nothing else. No explanation, no labels.
""".strip()


class SuperforecasterAgent(DeployableTraderAgent):
    """
    Two-stage prediction agent based on Berlin2OpenaiSearchAgentVariable,
    with superforecaster (Tetlock / Good Judgment Project) reasoning layered
    into both prompts.

    Stage 1 — gpt-4o with web_search_preview:
        Retrieves and organises evidence under superforecaster-relevant headings
        (base rates, reference classes, factors for/against, uncertainty).

    Stage 2 — o3-mini with high reasoning effort:
        Applies the full outside-view → inside-view → synthesis → bias-check
        framework to produce a calibrated probability and confidence score.
    """

    bet_on_n_markets_per_run = 2

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        return MaxAccuracyWithKellyScaledBetsStrategy(
            max_position_amount=get_maximum_possible_bet_amount(
                min_=USD(0.01),
                max_=USD(0.05),
                trading_balance=market.get_trade_balance(self.api_keys),
            ),
        )

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        client = OpenAI(api_key=self.api_keys.openai_api_key.get_secret_value())
        today = utcnow()

        # -------------------------------------------------------------------
        # Stage 1: Evidence gathering (gpt-4o + web search)
        # -------------------------------------------------------------------
        # The model retrieves and organises evidence under headings that
        # directly feed the superforecaster framework in stage 2.
        # Crucially, it is explicitly instructed NOT to draw conclusions —
        # that separation keeps the evidence report clean and unanchored.

        search_response = client.responses.create(
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
                    "content": SEARCH_DEVELOPER_PROMPT.format(today=today),
                },
                {
                    "role": "user",
                    "content": market.question,
                },
            ],
        )

        evidence_report = search_response.output_text

        # -------------------------------------------------------------------
        # Stage 2: Superforecaster reasoning (o3-mini, high effort)
        # -------------------------------------------------------------------
        # o3-mini with high reasoning effort is well-suited to the deliberate
        # multi-step structure of the Tetlock framework. We give it the full
        # outside-view → bias-check pipeline and let its internal chain-of-
        # thought handle the work. Output is kept to two floats to match the
        # original agent's contract.

        reasoning_response = client.responses.create(
            model="o3-mini",
            input=[
                {
                    "role": "developer",
                    "content": REASONING_DEVELOPER_PROMPT.format(today=today),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {market.question}\n\n"
                        f"Evidence report:\n{evidence_report}"
                    ),
                },
            ],
            reasoning={"effort": "high"},
        )

        raw_output = reasoning_response.output_text.strip()

        try:
            probability, confidence = map(float, raw_output.split())
        except Exception as e:
            raise UnexpectedModelBehavior(
                f"Could not parse probability and confidence from: '{raw_output}'"
            ) from e

        # Soft-clip as a safety net — the prompt instructs the model to stay
        # within 0.05–0.95, but we enforce it here in case it slips.
        probability = max(0.05, min(0.95, probability))
        confidence = max(0.0, min(1.0, confidence))

        return ProbabilisticAnswer(
            confidence=confidence,
            p_yes=Probability(probability),
        )
