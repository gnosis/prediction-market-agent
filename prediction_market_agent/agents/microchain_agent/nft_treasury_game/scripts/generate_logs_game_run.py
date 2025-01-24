import json
import typing as t

import typer
from prediction_market_agent_tooling.loggers import logger
from pydantic import BaseModel
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    NFT_TOKEN_FACTORY,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.deploy_nft_treasury_game import (
    DEPLOYED_NFT_AGENTS,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.scripts.fetch_metrics import (
    extract_transactions_involving_agents_and_treasuries,
    fetch_nft_transfers,
)
from prediction_market_agent.db.agent_communication import fetch_unseen_transactions


def main(rpc_url: str, write_output: bool = False) -> None:
    WRITE_OUTPUT = False
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    from_block = 38203034
    to_block = None
    transfers = fetch_nft_transfers(
        web3=w3,
        nft_contract_address=NFT_TOKEN_FACTORY,
        from_block=from_block,
        to_block=to_block,
    )

    messages = []
    for agent in DEPLOYED_NFT_AGENTS:
        messages.extend(
            fetch_unseen_transactions(consumer_address=agent.wallet_address, web3=w3)
        )

    transactions = extract_transactions_involving_agents_and_treasuries(
        web3=w3, from_block=from_block, to_block=to_block
    )
    if write_output:
        export_pydantic_models(transfers, "transfers")
        export_pydantic_models(messages, "messages")
        export_pydantic_models(transactions, "transactions")


def export_pydantic_models(
    items: t.Sequence[BaseModel],
    file_identifier: str,
    properties_to_exclude: t.Set[str] | None = None,
) -> None:
    properties_to_exclude = (
        set() if not properties_to_exclude else properties_to_exclude
    )
    filepath = f"{file_identifier}.json"
    logger.info(f"Writing {len(items)} items to {filepath}")
    with open(filepath, "w") as f:
        json.dump(
            [i.model_dump(exclude=properties_to_exclude) for i in items], f, indent=2
        )


if __name__ == "__main__":
    typer.run(main)
