from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer


class AnswerWithScenario(ProbabilisticAnswer):
    scenario: str
    original_question: str

    @staticmethod
    def build_from_probabilistic_answer(
        answer: ProbabilisticAnswer, scenario: str, question: str
    ) -> "AnswerWithScenario":
        return AnswerWithScenario(
            scenario=scenario, original_question=question, **answer.model_dump()
        )
