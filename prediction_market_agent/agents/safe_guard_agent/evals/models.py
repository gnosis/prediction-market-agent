from dataclasses import dataclass
from typing import TypeAlias

from prediction_market_agent_tooling.tools.utils import check_not_none
from pydantic import BaseModel
from pydantic_evals import Case
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from prediction_market_agent.agents.safe_guard_agent.safe_api_models.balances import (
    Balances,
)
from prediction_market_agent.agents.safe_guard_agent.safe_api_utils import (
    DetailedTransactionResponse,
)
from prediction_market_agent.agents.safe_guard_agent.safe_guard import (
    ValidationConclusion,
)

SGCase: TypeAlias = Case[
    tuple[DetailedTransactionResponse, Balances | None],
    ValidationConclusion,
    str,
]


class PydanticCase(BaseModel):
    """
    For some reason, Pydantic's Evals aren't BaseModel, so use this for json de/serialization.
    """

    name: str | None
    inputs: tuple[DetailedTransactionResponse, Balances | None]
    expected_output: ValidationConclusion | None
    metadata: str | None

    @staticmethod
    def from_case(
        case: SGCase,
    ) -> "PydanticCase":
        return PydanticCase(
            name=case.name,
            inputs=case.inputs,
            expected_output=case.expected_output,
            metadata=case.metadata,
        )

    def to_case(
        self,
    ) -> SGCase:
        return Case(
            name=self.name,
            inputs=self.inputs,
            expected_output=self.expected_output,
            metadata=self.metadata,
        )


@dataclass
class ValidationConclusionEvaluator(Evaluator):
    async def evaluate(
        self, ctx: EvaluatorContext[str, ValidationConclusion]
    ) -> EvaluationReason:
        expected_output = check_not_none(ctx.expected_output)
        incorrect_results = [
            res for res in ctx.output.results if res.ok != expected_output.all_ok
        ]

        if ctx.output.all_ok == expected_output.all_ok:
            return EvaluationReason(value=1.0, reason="All matches.")

        else:
            return EvaluationReason(
                value=0.0, reason=f"Incorrect results: {incorrect_results}"
            )
