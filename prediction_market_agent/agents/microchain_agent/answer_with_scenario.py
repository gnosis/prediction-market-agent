from prediction_market_agent_tooling.deploy.agent import Answer


class AnswerWithScenario(Answer):
    scenario: str
    question: str

    @staticmethod
    def build_from_answer(
        answer: Answer, scenario: str, question: str
    ) -> "AnswerWithScenario":
        return AnswerWithScenario(scenario=scenario, question=question, **answer.dict())
