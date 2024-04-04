import time
from datetime import timedelta, datetime

from dotenv import load_dotenv
from prediction_market_agent_tooling.benchmark.agents import AbstractBenchmarkedAgent
from prediction_market_agent_tooling.benchmark.benchmark import Benchmarker
from prediction_market_agent_tooling.benchmark.utils import (
    OutcomePrediction,
    Prediction, Market,
)
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.markets.markets import AgentMarket
from prediction_market_agent_tooling.tools.utils import utcnow
from pydantic import BaseModel

from prediction_market_agent.agents.crewai_subsequential_agent.crewai_agent_subquestions import CrewAIAgentSubquestions
from prediction_market_agent.agents.known_outcome_agent.known_outcome_agent import (
    Result,
    get_known_outcome,
)


def build_market_from_question_without_validation(question: str) -> Market:
    return Market.model_construct(url=question,question=question, p_yes = 0.5)


def build_binary_agent_market_from_question(question: str) -> AgentMarket:
    return AgentMarket(
        url="",
        id=question,
        question=question,
        p_yes=Probability(0.5),
        volume=None,
        created_time=datetime(2024,1,1),
        close_time=None,
        resolution=None,
        outcomes=["YES", "NO"],
    )

class QuestionAndAnswer(BaseModel):
    question: str
    result: Result
    bet_correct: bool



class CrewAIAgentSubquestionsBenchmark(AbstractBenchmarkedAgent):
    def __init__(
        self,
        agent_name: str,
        max_workers: int,
        model: str,
        max_tries: int,
    ) -> None:
        self.model = model
        self.max_tries = max_tries
        self.agent = CrewAIAgentSubquestions()
        super().__init__(agent_name=agent_name, max_workers=max_workers)


    def predict(self, market_question: str) -> Prediction:

        market = build_binary_agent_market_from_question(market_question)
        result = self.agent.answer_binary_market(market)

        answer = get_known_outcome(
            model=self.model,
            question=market_question,
            max_tries=self.max_tries,
        )
        print(f"Answered {market_question=} with {answer.result=}, {answer.reasoning=}")
        if not answer.has_known_result():
            return Prediction(
                is_predictable=False,
                outcome_prediction=None,
            )
        else:
            return Prediction(
                is_predictable=True,
                outcome_prediction=OutcomePrediction(
                    p_yes=answer.result.to_p_yes(),
                    confidence=1.0,
                    info_utility=None,
                ),
            )


if __name__ == "__main__":
    load_dotenv()
    tomorrow_str = (utcnow() + timedelta(days=1)).strftime("%d %B %Y")

    # Fetch example questions which our agents answered in the past.
    questions = [
        QuestionAndAnswer(
            question="Will the stock price of Donald Trump's media company exceed $100 on 1 April 2024?",
            result=Result.NO,
            bet_correct=True
        ),
        QuestionAndAnswer(
            question="Will Andy Murray return to professional tennis from his ankle injury on or before 31 March 2024?",
            result=Result.NO,
            bet_correct=True
        ),
        QuestionAndAnswer(
            question="Will any legislation be signed by President Biden that could potentially lead to the ban of TikTok by 1 April 2024?",
            result=Result.YES,
            bet_correct=False
        ),
        QuestionAndAnswer(
            question="Will the United States v. Apple case have a verdict by 1 April 2024?",
            result=Result.NO,
            bet_correct=True
        ),
        QuestionAndAnswer(
            question="Will Microsoft Teams launch the announced Copilot AI features by 1 April 2024?",
            result=Result.YES,
            bet_correct=True
        ),
        QuestionAndAnswer(
            question="Will the Francis Scott Key Bridge in Baltimore be fully rebuilt by 2 April 2024?",
            result=Result.NO,
            bet_correct=True
        ),
        QuestionAndAnswer(
            question="Will iOS 18 break the iPhone's iconic app grid by 1 April 2024?",
            result=Result.YES,
            bet_correct=False
        ),
        QuestionAndAnswer(
            question="Will a winner of the Mega Millions jackpot be announced by 26 March 2024?",
            result=Result.YES,
            bet_correct=False
        ),
    ]

    benchmarker = Benchmarker(
        markets=[build_market_from_question_without_validation(q.question) for q in questions][:1],
        agents=[
            CrewAIAgentSubquestionsBenchmark(
                agent_name="subsequential_questions",
                model="gpt-3.5-turbo-0125",
                max_tries=3,
                max_workers=1,
            ),
        ],
    )
    benchmarker.run_agents()
    md = benchmarker.generate_markdown_report()

    output = f"./subsequential_questions_agent_benchmark_report.{int(time.time())}.md"
    with open(output, "w") as f:
        print(f"Writing benchmark report to: {output}")
        f.write(md)

    # Check all predictions are correct, i.e. mean-squared-error == 0
    metrics = benchmarker.compute_metrics()
    assert metrics["MSE for `p_yes`"][0] == 0.0
