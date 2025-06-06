from typing import Sequence

import pandas as pd
import typer
from prediction_market_agent_tooling.gtypes import ChainID, ChecksumAddress
from tqdm import tqdm
from web3 import Web3

from prediction_market_agent.agents.safe_watch_agent.evals.models import (
    SGCase,
    SGDataset,
    ValidationConclusionEvaluator,
)
from prediction_market_agent.agents.safe_watch_agent.safe_api_utils import (
    get_balances_usd,
    get_safe_detailed_transaction_info,
    get_safe_history_multisig,
)
from prediction_market_agent.agents.safe_watch_agent.safe_watch import (
    ValidationConclusion,
)


def main(n: int = 10, chain_id: int = 100) -> None:
    """
    Gathers testing data that should all be okay.
    It's always latest transaction from biggest Safes that there are.
    """
    # Downloaded from https://dune.com/queries/4933998/8164678.
    data = pd.read_csv(
        "prediction_market_agent/agents/safe_watch_agent/evals/data/big_safes.csv"
    )
    big_safes = [Web3.to_checksum_address(addr) for addr in data["safe_address"]]

    cases = get_latest_transaction_cases(big_safes[:n], chain_id=ChainID(chain_id))

    evaluators = [ValidationConclusionEvaluator()]
    dataset = SGDataset(cases=cases, evaluators=evaluators)

    dataset.to_file(
        f"prediction_market_agent/agents/safe_watch_agent/evals/data/latest_transaction_cases.{chain_id}.yaml"
    )


def get_latest_transaction_cases(
    safe_addresses: list[ChecksumAddress], chain_id: ChainID
) -> Sequence[SGCase]:
    cases: list[SGCase] = []
    for safe_address in tqdm(safe_addresses):
        multisig_history = get_safe_history_multisig(
            safe_address=safe_address, chain_id=chain_id
        )
        if not multisig_history:
            continue
        newest_multisig_transaction = multisig_history[0]
        as_detailed = get_safe_detailed_transaction_info(
            newest_multisig_transaction.id, chain_id=chain_id
        )
        cases.append(
            SGCase(
                name=as_detailed.txId,
                inputs=(
                    chain_id,
                    as_detailed,
                    get_balances_usd(safe_address, chain_id=chain_id),
                ),
                expected_output=ValidationConclusion(
                    txId=as_detailed.txId, all_ok=True, summary="", results=[]
                ),
                metadata="Latest transaction that was processed for given Safe. Assumed to be OK.",
            )
        )

    return cases


if __name__ == "__main__":
    typer.run(main)
