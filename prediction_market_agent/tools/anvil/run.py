import json

import typer
from prediction_market_agent_tooling.loggers import logger
from pydantic import BaseModel
from web3 import Web3

from prediction_market_agent.tools.anvil.fetch_metrics import (
    extract_balances_per_block,
    fetch_nft_transfers,
    extract_messages_exchanged,
)


def main() -> None:
    WRITE_OUTPUT = True
    NFT_CONTRACT = Web3.to_checksum_address(
        "0x0D7C0Bd4169D090038c6F41CFd066958fe7619D0"
    )
    RPC_URL = "http://localhost:8545"
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    from_block = 38142651
    to_block = None
    ###############
    transfers = fetch_nft_transfers(web3=w3, nft_contract_address=NFT_CONTRACT)
    messages = extract_messages_exchanged(
        web3=w3, from_block=from_block, to_block=to_block
    )
    from_block = 38143011
    balances = extract_balances_per_block(
        RPC_URL, from_block=from_block, to_block=from_block + 10
    )
    if WRITE_OUTPUT:
        export_pydantic_models(transfers, "transfers")
        export_pydantic_models(messages, "messages", properties_to_exclude=["message"])
        export_pydantic_models(balances, "balances")

    print("finished")


def export_pydantic_models(
    items: list[BaseModel], file_identifier: str, properties_to_exclude=None
) -> None:
    properties_to_exclude = [] if not properties_to_exclude else properties_to_exclude
    filepath = f"{file_identifier}.json"
    logger.info(f"Writing {len(items)} items to {filepath}")
    with open(filepath, "w") as f:
        json.dump([i.model_dump(exclude=properties_to_exclude) for i in items], f)


if __name__ == "__main__":
    typer.run(main)
