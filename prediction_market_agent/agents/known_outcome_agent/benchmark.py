import time
import typing as t
from datetime import timedelta

from dotenv import load_dotenv
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.benchmark.agents import AbstractBenchmarkedAgent
from prediction_market_agent_tooling.benchmark.benchmark import Benchmarker
from prediction_market_agent_tooling.benchmark.utils import (
    OutcomePrediction,
    Prediction,
)
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.markets.markets import AgentMarket
from prediction_market_agent_tooling.tools.utils import utcnow
from pydantic import BaseModel

from prediction_market_agent.agents.known_outcome_agent.known_outcome_agent import (
    Result,
    get_known_outcome,
)


class QuestionWithKnownOutcome(BaseModel):
    url: t.Optional[str] = None
    question: str
    result: Result
    notes: t.Optional[str] = None

    def to_market(self) -> AgentMarket:
        return AgentMarket(
            url=self.url if self.url else "",
            id=self.question,
            question=self.question,
            current_p_yes=Probability(
                self.result.to_p_yes()
                if self.result != Result.KNOWN_UNKNOWABLE
                else 0.5
            ),
            volume=None,
            created_time=None,
            close_time=utcnow() + timedelta(days=1),
            resolution=None,
            outcomes=["YES", "NO"],
        )


class KnownOutcomeAgent(AbstractBenchmarkedAgent):
    def __init__(
        self,
        agent_name: str,
        max_workers: int,
        model: str,
        max_tries: int,
    ) -> None:
        self.model: str = model
        self.max_tries = max_tries
        super().__init__(agent_name=agent_name, max_workers=max_workers)

    def predict(self, market_question: str) -> Prediction:
        outcome = get_known_outcome(
            model=self.model,
            question=market_question,
            max_tries=self.max_tries,
        )
        logger.info(
            f"Answered {market_question=} with {outcome.result=}, {outcome.reasoning=}"
        )
        if not outcome.has_known_result():
            return Prediction(
                is_predictable=False,
                outcome_prediction=None,
            )
        else:
            return Prediction(
                is_predictable=True,
                outcome_prediction=OutcomePrediction(
                    decision=outcome.result.to_boolean(),
                    p_yes=outcome.result.to_p_yes(),
                    confidence=1.0,
                    info_utility=None,
                ),
            )


if __name__ == "__main__":
    load_dotenv()
    tomorrow_str = (utcnow() + timedelta(days=1)).strftime("%d %B %Y")

    # Fetch questions from existing markets, or make some up, where the
    # outcome is known.
    qs_with_known_outcome: list[QuestionWithKnownOutcome] = [
        QuestionWithKnownOutcome(
            question=f"Will 'Barbie' win an Academy Award for best original song by {tomorrow_str}?",
            url="https://aiomen.eth.limo/#/0xceb2a4ecc217cab440acf60737a9fcfd6d3fbf4b",
            result=Result.YES,
            notes="Happened on 10th March 2024.",
        ),
        QuestionWithKnownOutcome(
            question=f"Will the 2024 Oscars winner for Best Picture be announced by {tomorrow_str}?",
            url="https://aiomen.eth.limo/#/0xb88e4507709148e096bcdfb861b17db7b4d54e6b",
            result=Result.YES,
            notes="Happened on 10th March 2024.",
        ),
        QuestionWithKnownOutcome(
            question=f"Will Liverpool win against Atalanta in the Europa League quarter-finals by {tomorrow_str}?",
            url="https://aiomen.eth.limo/#/0x1d5a462c801360b4bebbda2b9656e52801a27a3b",
            result=Result.NO,
            notes="The match is scheduled for 11 April 2024.",
        ),
        QuestionWithKnownOutcome(
            question=f"Will Donald Trump officially become the GOP nominee for the 2024 presidential elections by {tomorrow_str}?",
            url="https://aiomen.eth.limo/#/0x859a6b465ee1e4a73aab0f2da4428c6255da466c",
            result=Result.YES,
            notes="Happened on 10th March 2024.",
        ),
        QuestionWithKnownOutcome(
            question=f"Will SpaceX successfully test a Starship reentry without losing contact by {tomorrow_str}?",
            url="https://aiomen.eth.limo/#/0xcc9123af8db309e0c60c63f9e2b8b82fc86f458b",
            result=Result.NO,
            notes="The only scheduled test flight occured, and contact was lost during the test.",
        ),
        QuestionWithKnownOutcome(
            question=f"Will Arsenal reach the Champions League semi-finals on {tomorrow_str}?",
            url="https://aiomen.eth.limo/#/0x606efd175b245cd60282a98cef402d4f5e950f92",
            result=Result.NO,
            notes="They are scheduled to play the first leg of the quarter-finals on 9 April 2024.",
        ),
        QuestionWithKnownOutcome(
            question=f"Will the jury deliver a verdict on James Crumbley's 'bad parenting' case on {tomorrow_str}?",
            url="https://aiomen.eth.limo/#/0xe55171beda0d60fd45092ff8bf93d5cb566a2510",
            result=Result.NO,
            notes="The verdict was announced on 15th March 2024.",
        ),
        QuestionWithKnownOutcome(
            question="Will Lewis Hamilton win the 2024/2025 F1 drivers champtionship?",
            result=Result.KNOWN_UNKNOWABLE,
            notes="Outcome is uncertain.",
        ),
        QuestionWithKnownOutcome(
            question="Will the cost of grain in the Spain increase by 20% by 19 July 2024?",
            result=Result.KNOWN_UNKNOWABLE,
            notes="Outcome is uncertain.",
        ),
        QuestionWithKnownOutcome(
            question="Will over 360 pople have died while climbing Mount Everest by 1st Jan 2028?",
            result=Result.KNOWN_UNKNOWABLE,
            notes="Outcome is uncertain.",
        ),
    ]

    benchmarker = Benchmarker(
        markets=[q.to_market() for q in qs_with_known_outcome],
        agents=[
            KnownOutcomeAgent(
                agent_name="known_outcome",
                model="gpt-4-1106-preview",
                max_tries=3,
                max_workers=1,
            ),
        ],
    )
    benchmarker.run_agents()
    md = benchmarker.generate_markdown_report()

    output = f"./known_outcome_agent_benchmark_report.{int(time.time())}.md"
    with open(output, "w") as f:
        logger.info(f"Writing benchmark report to: {output}")
        f.write(md)

    # Check all predictions are correct, i.e. mean-squared-error == 0
    metrics = benchmarker.compute_metrics()
    assert metrics["MSE for `p_yes`"][0] == 0.0
