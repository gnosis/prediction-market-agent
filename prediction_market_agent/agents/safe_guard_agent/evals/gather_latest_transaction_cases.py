import json

import pandas as pd
import typer
from prediction_market_agent_tooling.gtypes import ChecksumAddress
from pydantic_evals import Case
from tqdm import tqdm
from web3 import Web3

from prediction_market_agent.agents.safe_guard_agent.evals.models import (
    PydanticCase,
    SGCase,
)
from prediction_market_agent.agents.safe_guard_agent.safe_api_utils import (
    get_balances_usd,
    get_safe_detailed_transaction_info,
    get_safe_history_multisig,
)
from prediction_market_agent.agents.safe_guard_agent.safe_guard import (
    ValidationConclusion,
)


def main(n: int = 10) -> None:
    """
    Gathers testing data that should all be okay.
    It's always latest transaction from biggest Safes that there are.
    """
    # Downloaded from https://dune.com/queries/4933998/8164678.
    data = pd.read_csv(
        "prediction_market_agent/agents/safe_guard_agent/evals/data/big_safes.csv"
    )
    big_safes = [Web3.to_checksum_address(addr) for addr in data["safe_address"]]

    cases = get_latest_transaction_cases(big_safes[:n])

    with open(
        "prediction_market_agent/agents/safe_guard_agent/evals/data/latest_transaction_cases.json",
        "w",
    ) as f:
        json.dump([PydanticCase.from_case(x).model_dump() for x in cases], f, indent=2)


def get_latest_transaction_cases(safe_addresses: list[ChecksumAddress]) -> list[SGCase]:
    cases: list[SGCase] = []
    for safe_address in tqdm(safe_addresses):
        multisig_history = get_safe_history_multisig(safe_address=safe_address)
        if not multisig_history:
            continue
        newest_multisig_transaction = multisig_history[0]
        as_detailed = get_safe_detailed_transaction_info(newest_multisig_transaction.id)
        cases.append(
            Case(
                name=as_detailed.txId,
                inputs=(as_detailed, get_balances_usd(safe_address)),
                expected_output=ValidationConclusion(all_ok=True, results=[]),
                metadata="Latest transaction that was processed for given Safe. Assumed to be OK.",
            )
        )

    return cases


if __name__ == "__main__":
    typer.run(main)
