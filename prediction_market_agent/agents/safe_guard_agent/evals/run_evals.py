import json
from pathlib import Path
from unittest.mock import patch

import nest_asyncio
import typer
from prediction_market_agent_tooling.loggers import logger
from pydantic_evals import Dataset
from rich.console import Console

from prediction_market_agent.agents.safe_guard_agent.evals.models import (
    PydanticCase,
    SGCase,
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
    report_filename: str = "report.txt",
    data_folder: str = "prediction_market_agent/agents/safe_guard_agent/evals/data",
    limit: int | None = None,
) -> None:
    """
    Data folder must contain JSON files of serialized list[PydanticCase].
    Prepared eaither manually or using the scripts in this folder.
    """
    cases = load_cases(data_folder, limit)
    logger.info(f"Loaded {len(cases)} cases from {data_folder}.")

    evaluators = [ValidationConclusionEvaluator()]
    dataset = Dataset(cases=cases, evaluators=evaluators)

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


def load_cases(data_folder: str | Path, limit: int | None) -> list[SGCase]:
    cases: list[SGCase] = []
    for file in Path(data_folder).glob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)
            cases.extend(PydanticCase.model_validate(x).to_case() for x in data)
    if limit:
        cases = cases[:limit]
    return cases


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
