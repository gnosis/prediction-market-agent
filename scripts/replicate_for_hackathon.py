from datetime import timedelta

import typer
from prediction_market_agent_tooling.gtypes import USD, private_key_type
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.omen import (
    OMEN_DEFAULT_REALITIO_BOND_VALUE,
)
from prediction_market_agent_tooling.tools.utils import utcnow

from prediction_market_agent.agents.replicate_to_omen_agent.omen_replicate import (
    omen_replicate_from_tx,
)
from prediction_market_agent.agents.replicate_to_omen_agent.omen_resolve_replicated import (
    omen_finalize_and_resolve_and_claim_back_all_replicated_markets_tx,
)
from prediction_market_agent.utils import APIKeys


def main(
    from_private_key: (
        str | None
    ) = None,  # This is the private key of the address that will be used to create the markets, so people can filter the markets by this address.
    test: bool = True,  # Test is on by default, so we don't accidentally create the markets.
    n_to_replicate: int = 50,
    initial_funds_usd: float = 0.1,  # Just a small amount to make the markets bettable, probably no need for more in hackathon.
    resolve: bool = False,
) -> None:
    keys = APIKeys(
        BET_FROM_PRIVATE_KEY=(
            private_key_type(from_private_key) if from_private_key else None
        )
    )

    if not test and from_private_key is None:
        raise ValueError(
            "You need to provide private key for the real replication, based on the address promised at the hackathon."
        )

    if resolve:
        omen_finalize_and_resolve_and_claim_back_all_replicated_markets_tx(
            keys, realitio_bond=OMEN_DEFAULT_REALITIO_BOND_VALUE
        )
        return

    # Get participants at least 2 weeks to make the bets.
    close_time_after = utcnow() + timedelta(days=14)
    # But make sure that the markets won't be open too long, so we can evaluate results in a reasonable time.
    close_time_before = close_time_after + timedelta(days=7)

    addresses = omen_replicate_from_tx(
        api_keys=keys,
        market_type=MarketType.MANIFOLD,  # Use only Manifold, it has generally more markets.
        n_to_replicate=n_to_replicate,
        initial_funds=USD(initial_funds_usd),
        close_time_before=close_time_before,
        close_time_after=close_time_after,
        auto_deposit=True,
        test=test,
    )
    logger.info(f"Created {len(addresses)} markets: {addresses}")


if __name__ == "__main__":
    typer.run(main)
