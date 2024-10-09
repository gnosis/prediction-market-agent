from pprint import pprint

import typer
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.gtypes import private_key_type
from web3 import Web3

from prediction_market_agent.agents.replicate_to_omen_agent.deploy import (
    REPLICATOR_BOND,
)
from prediction_market_agent.agents.replicate_to_omen_agent.omen_resolve_replicated import (
    omen_finalize_and_resolve_and_claim_back_all_markets_based_on_others_tx,
)

# Use without the pretty exceptions, because they make the error stack unusable here.
app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    from_private_key: str = typer.Option(),
    safe_address: str = typer.Option(default=None),
) -> None:
    """
    Helper script to resolve markets on Omen that were replicated by the replication function.

    ```bash
    python scripts/resolve_replicated_on_omen.py --from-private-key your-private-key
    ```
    """
    safe_address_checksum = (
        Web3.to_checksum_address(safe_address) if safe_address else None
    )
    api_keys = APIKeys(
        BET_FROM_PRIVATE_KEY=private_key_type(from_private_key),
        SAFE_ADDRESS=safe_address_checksum,
    )

    result = omen_finalize_and_resolve_and_claim_back_all_markets_based_on_others_tx(
        api_keys=api_keys, realitio_bond=REPLICATOR_BOND
    )
    pprint(result.model_dump())


if __name__ == "__main__":
    app()
