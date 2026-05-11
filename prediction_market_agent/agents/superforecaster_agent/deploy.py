from openai import OpenAI
from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.deploy.betting_strategy import (
    BettingStrategy,
    MaxAccuracyWithKellyScaledBetsStrategy,
)
from prediction_market_agent_tooling.gtypes import USD, Probability
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.tools.relevant_news_analysis.relevant_news_analysis import (
    get_certified_relevant_news_since_cached,
)

from prediction_market_agent_tooling.tools.utils import utcnow
from pydantic_ai.exceptions import UnexpectedModelBehavior

from prediction_market_agent.agents.utils import get_maximum_possible_bet_amount

# How much p_yes must shift before we place an updated bet on an open position
REBET_THRESHOLD = 0.10


# Stage 1 prompt — evidence gathering, superforecaster-aware

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


# Stage 1 prompt variant — focused news update for re-evaluation

SEARCH_UPDATE_DEVELOPER_PROMPT = """
Today is {today}.

You will be given a prediction market question. New relevant news has been
detected since {since_date}. Your task is to find and summarize only the NEW
developments since that date that are relevant to the question.

Structure your report under these headings:

NEW DEVELOPMENTS
- What has happened since {since_date} that is directly relevant?
- Be specific: dates, named sources, concrete facts.

IMPACT ON PROBABILITY
- Does this new information make the YES outcome more likely, less likely,
  or is the impact unclear?
- Explain the directional reasoning without stating a final probability.

REMAINING UNCERTAINTY
- What is still unknown or unresolved after these developments?

Rules:
- Do not provide links — explain all evidence in full.
- Do not draw a conclusion or state a probability. That is done elsewhere.
- Focus only on what is NEW — do not rehash background already known.
""".strip()


# Stage 2 prompt — full superforecaster reasoning

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

# Stage 2 prompt variant — update reasoning anchored on previous probability

REASONING_UPDATE_DEVELOPER_PROMPT = """
Today is {today}.

You are a Superforecaster trained in Philip Tetlock's Good Judgment Project
methods. You previously estimated the probability for a prediction market
question. New relevant news has since emerged and you must decide whether to
revise your estimate.

Your previous probability estimate was: {previous_probability:.2f}

Work through these steps:

1. NEWS IMPACT
   - What does the new development concretely imply for the probability?
   - Is this a strong signal or weak/ambiguous evidence?

2. REVISION CHECK
   - Should you update significantly (>0.10 shift), modestly (0.05-0.10),
     or not at all (<0.05)?
   - Anchoring too hard on your previous estimate is a bias — correct for it
     if the news is genuinely informative.

3. BIAS CHECK
   - Are you over-reacting to vivid new information? (availability bias)
   - Are you under-reacting because you don't want to change your view?
     (belief perseverance)
   - Correct for whichever applies.

4. CONFIDENCE
   - Has your confidence increased or decreased given the new information?
   - Express as a float: 1.0 = near certainty, 0.0 = pure guess.

Calibration rules:
- Avoid probabilities below 0.05 or above 0.95.
- A single news item rarely justifies a revision of more than 0.20.
- If the news is ambiguous, stay close to your previous estimate.

Return ONLY two floats separated by a single space: probability confidence
Nothing else. No explanation, no labels.
""".strip()


# Core LLM helpers

def _run_two_stage_forecast(
    client: OpenAI,
    question: str,
    today: str,
) -> tuple[float, float]:
    """Full two-stage superforecaster forecast for a new market."""
    search_response = client.responses.create(
        model="gpt-4o",
        tools=[{"type": "web_search_preview", "search_context_size": "high"}],
        input=[
            {
                "role": "developer",
                "content": SEARCH_DEVELOPER_PROMPT.format(today=today),
            },
            {"role": "user", "content": question},
        ],
    )
    evidence_report = search_response.output_text

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
                    f"Question: {question}\n\n"
                    f"Evidence report:\n{evidence_report}"
                ),
            },
        ],
        reasoning={"effort": "medium"},
    )

    raw = reasoning_response.output_text.strip()
    try:
        probability, confidence = map(float, raw.split())
    except Exception as e:
        raise UnexpectedModelBehavior(
            f"Could not parse probability and confidence from: '{raw}'"
        ) from e

    return (
        max(0.05, min(0.95, probability)),
        max(0.0, min(1.0, confidence)),
    )


def _run_update_forecast(
    client: OpenAI,
    question: str,
    today: str,
    since_date: str,
    previous_probability: float,
) -> tuple[float, float]:
    """Focused update forecast for a market with new relevant news."""
    search_response = client.responses.create(
        model="gpt-4o",
        tools=[{"type": "web_search_preview", "search_context_size": "medium"}],
        input=[
            {
                "role": "developer",
                "content": SEARCH_UPDATE_DEVELOPER_PROMPT.format(
                    today=today,
                    since_date=since_date,
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    news_report = search_response.output_text

    reasoning_response = client.responses.create(
        model="o3-mini",
        input=[
            {
                "role": "developer",
                "content": REASONING_UPDATE_DEVELOPER_PROMPT.format(
                    today=today,
                    previous_probability=previous_probability,
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"New developments since {since_date}:\n{news_report}"
                ),
            },
        ],
        reasoning={"effort": "high"},
    )

    raw = reasoning_response.output_text.strip()
    try:
        probability, confidence = map(float, raw.split())
    except Exception as e:
        raise UnexpectedModelBehavior(
            f"Could not parse probability and confidence from: '{raw}'"
        ) from e

    return (
        max(0.05, min(0.95, probability)),
        max(0.0, min(1.0, confidence)),
    )


# Agent

class SuperforecasterAgent(DeployableTraderAgent):
    """
    Two-stage superforecaster agent with news-reactive position management.

    Each run has two phases:

    PHASE 1 — Re-evaluate open positions
        For each market this agent has previously bet on:
        - Check if relevant news has appeared since the last bet date
        - If yes: run a focused update forecast
        - If the new p_yes differs from the original by > REBET_THRESHOLD (0.10):
          place an updated bet in the new direction
        - If no news or shift too small: leave original bet untouched

    PHASE 2 — Bet on new markets (standard behavior)
        Pick up to bet_on_n_markets_per_run new markets and run the full
        two-stage superforecaster forecast on each.
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


    def _check_and_rebet_open_positions(self, market_type: MarketType) -> None:
        """Phase 1: scan open positions for relevant news and re-bet if warranted."""
        client = OpenAI(api_key=self.api_keys.openai_api_key.get_secret_value())
        today = utcnow()
        today_str = today.strftime("%A, %d %B %Y")
        user_id = self.api_keys.bet_from_address

        try:
            open_positions = OmenAgentMarket.get_positions(
                user_id=user_id,
                liquid_only=True,
                larger_than=USD(0.001),
            )
        except Exception as e:
            logger.warning(f"[Superforecaster] Could not fetch open positions: {e}")
            return

        if not open_positions:
            logger.info("[Superforecaster] No open positions to re-evaluate.")
            return

        logger.info(
            f"[Superforecaster] Checking {len(open_positions)} open positions "
            f"for relevant news."
        )

        for position in open_positions:
            try:
                market = OmenAgentMarket.get_binary_market(position.market_id)
            except Exception as e:
                logger.warning(
                    f"[Superforecaster] Could not fetch market "
                    f"{position.market_id}: {e}"
                )
                continue

            last_trade_datetime = market.get_most_recent_trade_datetime(
                user_id=user_id
            )
            if last_trade_datetime is None:
                continue

            days_since_last_trade = max((today - last_trade_datetime).days, 1)
            since_date_str = last_trade_datetime.strftime("%d %B %Y")

            logger.info(
                f"[Superforecaster] Checking '{market.question}' for news "
                f"since {since_date_str} ({days_since_last_trade}d ago)."
            )

            news = get_certified_relevant_news_since_cached(
                question=market.question,
                days_ago=days_since_last_trade,
                cache=None,
            )

            if news is None:
                logger.info(
                    f"[Superforecaster] No relevant news for "
                    f"'{market.question}' — skipping."
                )
                continue

            logger.info(
                f"[Superforecaster] Relevant news found for "
                f"'{market.question}' — running update forecast."
            )

            previous_probability = float(market.current_p_yes)

            try:
                new_probability, new_confidence = _run_update_forecast(
                    client=client,
                    question=market.question,
                    today=today_str,
                    since_date=since_date_str,
                    previous_probability=previous_probability,
                )
            except UnexpectedModelBehavior as e:
                logger.error(
                    f"[Superforecaster] Update forecast failed for "
                    f"'{market.question}': {e}"
                )
                continue

            shift = abs(new_probability - previous_probability)
            logger.info(
                f"[Superforecaster] '{market.question}': "
                f"prev={previous_probability:.3f} → new={new_probability:.3f} "
                f"(shift={shift:.3f}, confidence={new_confidence:.3f})"
            )

            if shift <= REBET_THRESHOLD:
                logger.info(
                    f"[Superforecaster] Shift {shift:.3f} below threshold "
                    f"{REBET_THRESHOLD} — no update bet."
                )
                continue

            logger.info(
                f"[Superforecaster] Placing update bet on '{market.question}'."
            )

            try:
                self.process_trade(
                    market=market,
                    answer=ProbabilisticAnswer(
                        p_yes=Probability(new_probability),
                        confidence=new_confidence,
                        reasoning=(
                            f"News-reactive update: {previous_probability:.2f} → "
                            f"{new_probability:.2f} after news since {since_date_str}."
                        ),
                    ),
                    market_type=market_type,
                )
            except Exception as e:
                logger.error(
                    f"[Superforecaster] Update bet failed for "
                    f"'{market.question}': {e}"
                )

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        """Phase 2: standard full forecast for new markets."""
        client = OpenAI(api_key=self.api_keys.openai_api_key.get_secret_value())
        today_str = utcnow().strftime("%A, %d %B %Y %H:%M UTC")

        logger.info(
            f"[Superforecaster] Full forecast for '{market.question}'."
        )

        try:
            probability, confidence = _run_two_stage_forecast(
                client=client,
                question=market.question,
                today=today_str,
            )
        except UnexpectedModelBehavior as e:
            logger.error(f"[Superforecaster] Forecast failed: {e}")
            return None

        logger.info(
            f"[Superforecaster] '{market.question}': "
            f"p_yes={probability:.3f}, confidence={confidence:.3f}"
        )

        return ProbabilisticAnswer(
            p_yes=Probability(probability),
            confidence=confidence,
        )

    def run(self, market_type: MarketType) -> None:
        """
        Override run to inject Phase 1 (news re-evaluation) before the
        standard Phase 2 (new market betting) loop.
        """
        logger.info("[Superforecaster] Phase 1: re-evaluating open positions.")
        self._check_and_rebet_open_positions(market_type)

        logger.info("[Superforecaster] Phase 2: betting on new markets.")
        super().run(market_type)