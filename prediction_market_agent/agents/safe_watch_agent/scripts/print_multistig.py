from pprint import pprint

import typer
from prediction_market_agent_tooling.gtypes import ChainID

from prediction_market_agent.agents.safe_watch_agent.safe_api_utils import (
    get_safe_detailed_transaction_info,
)
from prediction_market_agent.agents.safe_watch_agent.watchers.llm import format_balances


def main(tx_id: str, chain_id: int) -> None:
    """
    Helper script to print all information about multisig with given id from Safe APIs.
    """
    tx = get_safe_detailed_transaction_info(tx_id, chain_id=ChainID(chain_id))

    print(f"Safe's info:")
    print(format_balances(tx.safeAddress, chain_id=ChainID(chain_id)))
    print(f"Transaction info:")
    pprint(tx.model_dump())


if __name__ == "__main__":
    typer.run(main)
