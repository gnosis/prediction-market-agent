import re
from itertools import combinations
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.deploy.betting_strategy import (
    BettingStrategy,
    MaxAccuracyWithKellyScaledBetsStrategy,
)
from prediction_market_agent_tooling.gtypes import USD, Probability
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from pydantic import BaseModel

from prediction_market_agent.agents.utils import get_maximum_possible_bet_amount



# Constants

# Minimum shared meaningful tokens to consider two markets as pair candidates
MIN_KEYWORD_OVERLAP = 2

# Minimum deviation from expected sum to flag a pair as mispriced
# e.g. COMPLEMENTARY: |p_a + p_b - 1.0| > threshold
# e.g. MUTUALLY_EXCLUSIVE: p_a + p_b - 1.0 > threshold
MISPRICING_THRESHOLD = 0.06

# Stop words excluded from keyword overlap scoring
STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "will", "be", "by", "with", "from", "that", "this", "it", "its",
    "have", "has", "had", "are", "was", "were", "do", "does", "did",
    "would", "could", "should", "may", "might", "shall", "can",
    "not", "no", "yes", "if", "than", "then", "so", "as", "up",
    "before", "after", "during", "between", "into", "through",
    "who", "what", "when", "where", "which", "how", "any", "all",
}

# Data models

class PairRelationship(BaseModel):
    """Structured result of the LLM pair analysis."""
    relationship: Literal["COMPLEMENTARY", "MUTUALLY_EXCLUSIVE", "INDEPENDENT"]
    mispriced: bool
    bet_market: Literal["A", "B", "NONE"]
    bet_direction: Literal["YES", "NO", "NONE"]
    expected_sum: float
    actual_sum: float
    reasoning: str

# Prompts

PAIR_ANALYSIS_PROMPT = """
You are a prediction market analyst. You are given two open binary markets
and their current YES probabilities. Determine whether they are logically
related in a way that creates a pricing constraint, and if so, whether that
constraint is currently violated.

Market A: "{question_a}"
Current P(YES) for A: {prob_a:.3f}

Market B: "{question_b}"
Current P(YES) for B: {prob_b:.3f}

---

STEP 1 — Classify the relationship:

COMPLEMENTARY
Exactly one of A or B must resolve YES — they are two sides of the same
coin (e.g. "Will X win?" vs "Will X lose?" in a direct head-to-head).
Constraint: P(A) + P(B) should be ~1.0
Mispriced if |P(A) + P(B) - 1.0| > {threshold:.2f}

MUTUALLY_EXCLUSIVE
At most one can resolve YES, but both could resolve NO (e.g. two candidates
in a multi-horse race, two conflicting outcomes where neither may occur).
Constraint: P(A) + P(B) should be <= 1.0
Mispriced if P(A) + P(B) > 1.0 + {threshold:.2f}

INDEPENDENT
No meaningful pricing constraint. This is the default — only use
COMPLEMENTARY or MUTUALLY_EXCLUSIVE if you are confident in the relationship.

---

STEP 2 — If mispriced, identify the single best bet:

For COMPLEMENTARY: bet YES on the underpriced market (the one where
its probability is too low relative to 1 - P(other)).

For MUTUALLY_EXCLUSIVE: bet NO on the market with the higher probability,
as that is the most likely overpriced side.

If INDEPENDENT or not mispriced, set BET_MARKET and BET_DIRECTION to NONE.

---

STEP 3 — Brief reasoning (2-3 sentences max).

Respond in EXACTLY this format, nothing else:
RELATIONSHIP: <COMPLEMENTARY | MUTUALLY_EXCLUSIVE | INDEPENDENT>
MISPRICED: <YES | NO>
BET_MARKET: <A | B | NONE>
BET_DIRECTION: <YES | NO | NONE>
EXPECTED_SUM: <float>
ACTUAL_SUM: <float>
REASONING: <your reasoning>
""".strip()


# Keyword overlap pre-filter

def _tokenize(question: str) -> set[str]:
    tokens = re.findall(r"[a-z]+", question.lower())
    return {t for t in tokens if t not in STOP_WORDS and len(t) > 2}


def _keyword_overlap(q1: str, q2: str) -> int:
    return len(_tokenize(q1) & _tokenize(q2))


def find_candidate_pairs(
    markets: list[AgentMarket],
    min_overlap: int = MIN_KEYWORD_OVERLAP,
) -> list[tuple[AgentMarket, AgentMarket]]:
    """
    Pre-filter: returns market pairs sharing at least `min_overlap` meaningful
    tokens. Much cheaper than calling the LLM on every possible combination.
    """
    return [
        (m1, m2)
        for m1, m2 in combinations(markets, 2)
        if _keyword_overlap(m1.question, m2.question) >= min_overlap
    ]


# LLM pair analysis

def analyze_pair(
    market_a: AgentMarket,
    market_b: AgentMarket,
) -> PairRelationship | None:
    """
    Calls gpt-4o-mini to confirm whether a keyword-overlapping pair is
    logically related and mispriced. Returns None on parse failure.
    gpt-4o-mini is intentional here — this is called O(n²) times so cost
    and latency matter more than raw reasoning power.
    """
    prob_a = float(market_a.current_p_yes)
    prob_b = float(market_b.current_p_yes)

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        openai_api_key=APIKeys().openai_api_key,
        temperature=0,
    )

    prompt = ChatPromptTemplate([("user", PAIR_ANALYSIS_PROMPT)])
    messages = prompt.format_messages(
        question_a=market_a.question,
        prob_a=prob_a,
        question_b=market_b.question,
        prob_b=prob_b,
        threshold=MISPRICING_THRESHOLD,
    )

    response = llm.invoke(messages, max_tokens=300)
    raw = str(response.content)

    def extract(pattern: str) -> str | None:
        m = re.search(pattern, raw, re.IGNORECASE)
        return m.group(1).strip() if m else None

    relationship = extract(r"RELATIONSHIP:\s*(COMPLEMENTARY|MUTUALLY_EXCLUSIVE|INDEPENDENT)")
    mispriced_str = extract(r"MISPRICED:\s*(YES|NO)")
    bet_market = extract(r"BET_MARKET:\s*(A|B|NONE)")
    bet_direction = extract(r"BET_DIRECTION:\s*(YES|NO|NONE)")
    expected_sum_str = extract(r"EXPECTED_SUM:\s*([0-9.]+)")
    actual_sum_str = extract(r"ACTUAL_SUM:\s*([0-9.]+)")
    reasoning = extract(r"REASONING:\s*(.+)")

    if not all([relationship, mispriced_str, bet_market, bet_direction]):
        logger.warning(f"[Contradiction] Could not fully parse pair analysis:\n{raw}")
        return None

    return PairRelationship(
        relationship=relationship,  # type: ignore[arg-type]
        mispriced=(mispriced_str == "YES"),
        bet_market=bet_market,  # type: ignore[arg-type]
        bet_direction=bet_direction,  # type: ignore[arg-type]
        expected_sum=float(expected_sum_str) if expected_sum_str else 1.0,
        actual_sum=float(actual_sum_str) if actual_sum_str else prob_a + prob_b,
        reasoning=reasoning or "",
    )


# Agent

class ContradictionAgent(DeployableTraderAgent):
    """
    Scans open markets for logically related pairs whose combined probabilities
    violate a pricing constraint (complementary or mutually exclusive), then
    bets on the single highest-confidence correction per pair.

    Pipeline per run:
      1. load() fetches all open markets and runs the full pair scan upfront.
      2. Cheap keyword overlap pre-filter reduces LLM calls to plausible pairs only.
      3. gpt-4o-mini confirms the relationship and identifies the best single bet.
      4. Pairs are ranked by deviation magnitude; best bets stored in bet_lookup.
      5. answer_binary_market() checks the lookup and returns the queued answer.

    Design decisions:
      - Saturated markets are NOT skipped — a market at 0.95 can still be half
        of a mispriced complementary pair if its partner is also at 0.85.
      - Only the higher-confidence side of each pair is bet, so no cross-run
        state management is needed.
      - Confidence is derived from deviation size: larger mispricing = higher
        confidence the market will correct.
      - gpt-4o-mini for pair analysis (O(n²) calls, cost-sensitive).
      - Conservative max bet size since this is a structural/arbitrage play,
        not a directional forecast.
    """

    bet_on_n_markets_per_run = 2

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        return MaxAccuracyWithKellyScaledBetsStrategy(
            max_position_amount=get_maximum_possible_bet_amount(
                min_=USD(0.5),
                max_=USD(3.0),
                trading_balance=market.get_trade_balance(self.api_keys),
            ),
        )

    def load(self) -> None:
        """
        Pre-scans all open markets to build bet_lookup before the per-market
        betting loop begins. Stored as:
            { market_id: (p_yes, confidence, reasoning) }
        """
        self.bet_lookup: dict[str, tuple[float, float, str]] = {}

        try:
            open_markets = self.get_markets(market_type=self.supported_markets[0])
        except Exception as e:
            logger.error(f"[Contradiction] Could not fetch open markets in load(): {e}")
            return

        if len(open_markets) < 2:
            logger.info("[Contradiction] Fewer than 2 open markets — nothing to pair.")
            return

        logger.info(
            f"[Contradiction] Scanning {len(open_markets)} markets "
            f"({len(open_markets) * (len(open_markets) - 1) // 2} possible pairs)..."
        )

        candidates = find_candidate_pairs(open_markets)
        logger.info(
            f"[Contradiction] {len(candidates)} candidate pairs after keyword filter "
            f"(min_overlap={MIN_KEYWORD_OVERLAP})."
        )

        # Collect all mispriced pairs as (deviation, market, p_yes, confidence, reasoning)
        mispriced: list[tuple[float, AgentMarket, float, float, str]] = []

        for market_a, market_b in candidates:
            result = analyze_pair(market_a, market_b)

            if result is None or not result.mispriced or result.bet_market == "NONE":
                continue

            deviation = abs(result.actual_sum - result.expected_sum)
            target = market_a if result.bet_market == "A" else market_b
            p_yes = 1.0 if result.bet_direction == "YES" else 0.0

            # Confidence scales with mispricing magnitude, capped at 0.95
            confidence = min(0.5 + deviation * 2.0, 0.95)

            logger.info(
                f"[Contradiction] Mispriced pair found!\n"
                f"  A: '{market_a.question}' P={market_a.current_p_yes:.3f}\n"
                f"  B: '{market_b.question}' P={market_b.current_p_yes:.3f}\n"
                f"  Relationship : {result.relationship}\n"
                f"  Expected sum : {result.expected_sum:.3f}\n"
                f"  Actual sum   : {result.actual_sum:.3f}  (deviation={deviation:.3f})\n"
                f"  Best bet     : {result.bet_direction} on market {result.bet_market}\n"
                f"  Confidence   : {confidence:.3f}\n"
                f"  Reasoning    : {result.reasoning}"
            )

            mispriced.append((deviation, target, p_yes, confidence, result.reasoning))

        # Rank by deviation — biggest mispricing gets first pick
        mispriced.sort(key=lambda x: x[0], reverse=True)

        for _, market, p_yes, confidence, reasoning in mispriced:
            if market.id not in self.bet_lookup:
                self.bet_lookup[market.id] = (p_yes, confidence, reasoning)

        logger.info(
            f"[Contradiction] {len(self.bet_lookup)} market(s) queued for betting this run."
        )

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        if market.id not in self.bet_lookup:
            logger.info(
                f"[Contradiction] '{market.question}' — "
                "not part of any mispriced pair this run, skipping."
            )
            return None

        p_yes, confidence, reasoning = self.bet_lookup[market.id]

        logger.info(
            f"[Contradiction] Betting on '{market.question}' — "
            f"p_yes={p_yes:.2f}, confidence={confidence:.3f}"
        )

        return ProbabilisticAnswer(
            p_yes=Probability(p_yes),
            confidence=confidence,
            reasoning=reasoning,
        )