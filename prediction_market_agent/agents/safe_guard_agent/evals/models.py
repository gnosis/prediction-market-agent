from dataclasses import dataclass
from typing import TypeAlias

from prediction_market_agent_tooling.tools.utils import check_not_none
from pydantic_evals import Case, Dataset
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
SGDataset: TypeAlias = Dataset[
    tuple[DetailedTransactionResponse, Balances | None],
    ValidationConclusion,
    str,
]


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
