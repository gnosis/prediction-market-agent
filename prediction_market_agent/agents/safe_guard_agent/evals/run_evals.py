from unittest.mock import patch

import nest_asyncio
import typer
from prediction_market_agent_tooling.loggers import logger
from rich.console import Console

from prediction_market_agent.agents.safe_guard_agent.evals.models import (
    SGDataset,
    ValidationConclusionEvaluator,
)
from prediction_market_agent.agents.safe_guard_agent.safe_api_models.balances import (
    Balances,
)
from prediction_market_agent.agents.safe_guard_agent.safe_api_utils import (
    DetailedTransactionResponse,
    get_balances_usd,
)
from prediction_market_agent.agents.safe_guard_agent.safe_guard import (
    ValidationConclusion,
    validate_safe_transaction_obj,
)

nest_asyncio.apply()


def main(
    dataset_file: str,
    report_filename: str = "report.txt",
    limit: int | None = None,
) -> None:
    """
    Data folder must contain YAML files with Pydantic Evals Datasets.
    Prepared eaither manually or using the scripts in this folder.
    """
    dataset = SGDataset.from_file(
        dataset_file, custom_evaluator_types=[ValidationConclusionEvaluator]
    )
    if limit:
        dataset.cases = dataset.cases[:limit]
    logger.info(f"Loaded {len(dataset.cases)} cases from {dataset_file}.")

    report = dataset.evaluate_sync(process_case, max_concurrency=1)
    table = report.console_table(
        include_input=False,
        include_metadata=True,
        include_expected_output=True,
        include_output=True,
        include_durations=True,
        include_total_duration=True,
        include_removed_cases=True,
        include_averages=True,
    )

    console = Console(width=1600, record=True)
    console.print(table, crop=False)
    console.save_text(report_filename)


async def process_case(
    inputs: tuple[DetailedTransactionResponse, Balances | None],
) -> ValidationConclusion:
    # Pydantic eval doesn't allow multiple arguments, so we need to unpack it here.
    tx, balances_patch = inputs
    # In case balances_patch was provided, we need to use it instead of the real balances.
    # Handy when dealing with historical data, and the current balances aren't representative (and it confuses LLM).
    with patch(
        "prediction_market_agent.agents.safe_guard_agent.guards.llm.get_balances_usd",
        new=get_balances_usd if balances_patch is None else lambda x: balances_patch,
    ):
        result = validate_safe_transaction_obj(
            detailed_transaction_info=tx,
            do_sign_or_execution=False,
            do_reject=False,
            do_message=False,
            ignore_historical_transaction_ids={tx.txId},
        )
    return result


if __name__ == "__main__":
    typer.run(main)
