from prediction_market_agent_tooling.benchmark.agents import AbstractBenchmarkedAgent
from prediction_market_agent_tooling.benchmark.utils import Prediction
from prediction_market_agent_tooling.markets.data_models import (
    CategoricalProbabilisticAnswer,
)
from prediction_market_agent_tooling.gtypes import Probability

from prediction_market_agent.agents.multi_persona_agent.deploy import MultiPersonaAgent
from prediction_market_agent.agents.multi_persona_ensemble_agent.deploy import (
    MultiPersonaEnsembleAgent,
)


class MultiPersonaBenchmarkedAgent(AbstractBenchmarkedAgent):
    def __init__(self):
        super().__init__(agent_name="multi_persona_agent", max_workers=2)
        self.agent = MultiPersonaAgent()

    def predict(self, market_question: str) -> Prediction:
        # Fake minimal market object behavior
        class DummyMarket:
            question = market_question
            p_yes = 0.5

        answer = self.agent.answer_binary_market(DummyMarket())

        if answer is None:
            return Prediction(is_predictable=True)

        return Prediction(
            outcome_prediction=CategoricalProbabilisticAnswer.from_probabilistic_answer(
                answer
            )
        )


class MultiPersonaEnsembleBenchmarkedAgent(AbstractBenchmarkedAgent):
    def __init__(self):
        super().__init__(agent_name="multi_persona_ensemble_agent", max_workers=2)
        self.agent = MultiPersonaEnsembleAgent()

    def predict(self, market_question: str) -> Prediction:
        class DummyMarket:
            question = market_question
            p_yes = 0.5

        answer = self.agent.answer_binary_market(DummyMarket())

        if answer is None:
            return Prediction(is_predictable=True)

        return Prediction(
            outcome_prediction=CategoricalProbabilisticAnswer.from_probabilistic_answer(
                answer
            )
        )
