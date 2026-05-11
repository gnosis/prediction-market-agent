"""
Benchmark script for SuperforecasterAgent.

Usage:
    python prediction_market_agent/agents/superforecaster_agent/benchmark.py

Evaluates the agent against a set of historical questions with known outcomes.
To avoid data leakage, use questions that resolved AFTER the model's training
cutoff (roughly mid-2024 for gpt-4o / o3-mini).

Good sources for test questions:
- Metaculus: https://metaculus.com/questions (filter by resolved)
- Manifold: https://manifold.markets (filter by resolved)
- Polymarket: past resolved markets

Metrics computed:
- MSE on p_yes (lower is better, 0.0 = perfect)
- Brier score (same as MSE for binary outcomes)
- Mean confidence
- Accuracy (% of markets where agent predicted the correct outcome)
"""

import time
from datetime import timedelta

from dotenv import load_dotenv
from prediction_market_agent_tooling.benchmark.agents import AbstractBenchmarkedAgent
from prediction_market_agent_tooling.benchmark.benchmark import Benchmarker
from prediction_market_agent_tooling.benchmark.utils import Prediction
from prediction_market_agent_tooling.deploy.constants import (
    YES_OUTCOME_LOWERCASE_IDENTIFIER,
    NO_OUTCOME_LOWERCASE_IDENTIFIER,
)
from prediction_market_agent_tooling.gtypes import OutcomeStr, Probability
from prediction_market_agent_tooling.markets.data_models import (
    CategoricalProbabilisticAnswer,
    ProbabilisticAnswer,
)
from prediction_market_agent_tooling.markets.market_fees import MarketFees
from prediction_market_agent_tooling.markets.markets import AgentMarket
from prediction_market_agent_tooling.tools.utils import utcnow
from pydantic import BaseModel

from prediction_market_agent.agents.superforecaster_agent.deploy import (
    SuperforecasterAgent,
)


# ---------------------------------------------------------------------------
# Test questions
# ---------------------------------------------------------------------------
# Add more questions here as you find them. Each needs:
#   - question: the exact market question
#   - result: True if resolved YES, False if resolved NO
#   - source: optional, just for your reference
#
# IMPORTANT: Only use questions that resolved AFTER ~June 2024 to avoid
# the model having the answer in its training data (data leakage).
# ---------------------------------------------------------------------------

class TestQuestion(BaseModel):
    question: str
    result: bool
    source: str = ""


TEST_QUESTIONS: list[TestQuestion] = [

    # --- 2025 Events ---
    TestQuestion(
        question="Will Donald Trump be inaugurated as US President in January 2025?",
        result=True,
        source="Inaugurated Jan 20 2025",
    ),
    TestQuestion(
        question="Will the US enter a recession in the first half of 2025?",
        result=False,
        source="No recession declared H1 2025",
    ),

    # --- Add more questions below ---
    TestQuestion(
        question="Will the United States attack Iran before April 2026?",
        result=True,
        source="US attacked Iran in March 2026",
    ),

    TestQuestion(
        question="Will the UK Abolish the Two-Child Benefit Cap before 2035?",
        result=True,
        source="UK abolished this year in 2026",
    ),

    TestQuestion(
        question="Will a bot be #1 in the Metaculus 2026 Baseline Leaderboard on the Ides of March 2026?",
        result=False,
        source="Human was #1, closest bot was #2",
    ),

    TestQuestion(
        question="Will layoffs.fyi explicity report at least 100 AI industry layoffs between January 12 and March 13, 2026?",
        result=True,
        source="At least 280 layoffs reported in that period",
    ),

    TestQuestion(
        question="Will the United States impose additional sanctions on Russia related to the Ukraine war before March 14, 2026?",
        result=False,
        source="No new/expanded sanctions in this time frame, and US actually lifted some in this period",
    ),

    TestQuestion(
        question="Will Wagner Moura win the Academy Award for Best Actor at the 2026 Academy Awards?",
        result=False,
        source="Wagner Moura did not win the Academy Award for Best Actor in 2026, this went to Michael B Jordan",
    ),

    TestQuestion(
        question="Will OpenAI's ChatGPT Atlas browser be released for Windows before March 14, 2026?",
        result=False,
        source="Atlas was not released on Windows by this date",
    ),

    TestQuestion(
        question="Will Brent crude oil reach $100 a barrel before April 1, 2026?",
        result=True,
        source="Brent crude oil reached $100 a barrel in early/mid-March 2026",
    ),


    TestQuestion(
        question="Will Senator John Cornyn avoid elimination in the first round of the Texas Republican primary?",
        result=True,
        source="Senator John Cornyn advanced to a runoff in the Texas Republican primary, avoiding elimination in the first round.",
    ),


    TestQuestion(
        question="Will the number of manufacturing jobs in the US for February 2026 be above 12.7 million?",
        result=False,
        source="Number of manufacturing jobs was 12.573 million for February 2026",
    ),


    TestQuestion(
        question="Will Donald Trump say 'AI' or 'artificial intelligence' in his State of the Union address on February 24, 2026?",
        result=True,
        source="Trump said 'AI' in his address",
    ),

    TestQuestion(
        question="Will the U.S. Congress still be operating under a continuing resolution (CR) on March 15, 2026?",
        result=False,
        source="Congress was not operating under a CR on March 15, 2026",
    ),

    TestQuestion(
        question="Will the retail price of rice in Japan fall below ¥4200/5kg before March 9, 2026?",
        result=True,
        source="Price was ¥4095/5kg for the week of January 19, 2026",
    ),

    TestQuestion(
        question="Will the German 10-Year Bond Yield move by 20bps or more during February 2026?",
        result=False,
        source="Did not move by 20bps or more during February 2026",
    ),

    TestQuestion(
        question="In February 2026, will the weekly amount of finished motor gasoline supplied in the U.S. be highest in the final week of the month?",
        result=False,
        source="Weekly amount of finished motor gasoline supplied in the US in February 2026 was not highest in the final week of the month",
    ),

    TestQuestion(
        question="Will China request a WTO panel in its EV and battery-related trade dispute with India before January 11, 2026?",
        result=False,
        source="China did not request a WTO panel in this dispute before January 11, 2026",
    ),

    TestQuestion(
        question="Will the United States gain less than 100,000 new nonfarm jobs between December 2025 and February 2026?",
        result=True,
        source="US only gained about 30,000-40,000 new nonfarm jobs in that period",
    ),

    TestQuestion(
        question="Will there be at least one podium sweep at the 2026 Winter Olympic Games?",
        result=True,
        source="There was at least one podium sweep (Sweden in Womens' skating) at the 2026 Winter Olympic Games",
    ),


]


# ---------------------------------------------------------------------------
# Benchmarked agent wrapper
# ---------------------------------------------------------------------------

class BenchmarkedSuperforecaster(AbstractBenchmarkedAgent):
    def __init__(self, agent_name: str, max_workers: int) -> None:
        super().__init__(agent_name=agent_name, max_workers=max_workers)
        self.agent = SuperforecasterAgent()

    def predict(self, market_question: str) -> Prediction:
        dummy_market = AgentMarket(
            id=market_question,
            question=market_question,
            description=None,
            url="",
            probabilities={
                OutcomeStr(YES_OUTCOME_LOWERCASE_IDENTIFIER): Probability(0.5),
                OutcomeStr(NO_OUTCOME_LOWERCASE_IDENTIFIER): Probability(0.5),
            },
            volume=None,
            created_time=None,
            close_time=utcnow() + timedelta(days=1),
            resolution=None,
            outcomes=[
                OutcomeStr(YES_OUTCOME_LOWERCASE_IDENTIFIER),
                OutcomeStr(NO_OUTCOME_LOWERCASE_IDENTIFIER),
            ],
            outcome_token_pool=None,
            fees=MarketFees.get_zero_fees(),
        )

        answer: ProbabilisticAnswer | None = self.agent.answer_binary_market(
            dummy_market
        )

        if answer is None:
            return Prediction(is_predictable=False, outcome_prediction=None)

        return Prediction(
            is_predictable=True,
            outcome_prediction=CategoricalProbabilisticAnswer.from_probabilistic_answer(
                answer
            ),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_dotenv()

    markets = [
        AgentMarket(
            id=q.question,
            question=q.question,
            description=None,
            url=q.source,
            probabilities={
                OutcomeStr(YES_OUTCOME_LOWERCASE_IDENTIFIER): Probability(
                    1.0 if q.result else 0.0
                ),
                OutcomeStr(NO_OUTCOME_LOWERCASE_IDENTIFIER): Probability(
                    0.0 if q.result else 1.0
                ),
            },
            volume=None,
            created_time=None,
            close_time=utcnow() + timedelta(days=1),
            resolution=None,
            outcomes=[
                OutcomeStr(YES_OUTCOME_LOWERCASE_IDENTIFIER),
                OutcomeStr(NO_OUTCOME_LOWERCASE_IDENTIFIER),
            ],
            outcome_token_pool=None,
            fees=MarketFees.get_zero_fees(),
        )
        for q in TEST_QUESTIONS
    ]

    benchmarker = Benchmarker(
        markets=markets,
        agents=[
            BenchmarkedSuperforecaster(
                agent_name="superforecaster",
                max_workers=1,  # keep at 1 to avoid hammering the OpenAI API
            ),
        ],
    )

    print(f"Running benchmark on {len(TEST_QUESTIONS)} questions...")
    benchmarker.run_agents()

    md = benchmarker.generate_markdown_report()
    output = f"./superforecaster_benchmark_{int(time.time())}.md"
    with open(output, "w") as f:
        f.write(md)

    print(f"\nReport written to: {output}")
    print("\nMetrics:")
    metrics = benchmarker.compute_metrics()
    for k, v in metrics.items():
        print(f"  {k}: {v}")