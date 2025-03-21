import time

import typer
from prediction_market_agent_tooling.gtypes import private_key_type, xDai
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.omen.omen import (
    redeem_from_all_user_positions,
)
from prediction_market_agent_tooling.markets.omen.omen_contracts import (
    WrappedxDaiContract,
)
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    SAFE_COLLATERAL_TOKENS,
)
from prediction_market_agent_tooling.tools.balances import get_balances
from prediction_market_agent_tooling.tools.omen.sell_positions import sell_all
from web3 import Web3

from prediction_market_agent.agents.replicate_to_omen_agent.omen_resolve_replicated import (
    claim_all_bonds_on_reality,
)
from prediction_market_agent.utils import APIKeys


def main(
    from_private_key: str,
    to_address_: str,
    from_safe_address: str | None = None,
) -> None:
    """
    This script will transfer all resources that we usually work with from one address to another.
    If Safe is provided, it will transfer resources from there, instead of the EOA.
    Use for example in the case of leaked private key.
    """
    api_keys = APIKeys(
        BET_FROM_PRIVATE_KEY=private_key_type(from_private_key),
        SAFE_ADDRESS=(
            Web3.to_checksum_address(from_safe_address) if from_safe_address else None
        ),
    )
    to_address = Web3.to_checksum_address(to_address_)

    if api_keys.bet_from_address == to_address:
        raise ValueError("The addresses are the same.")

    logger.info(
        f"Going to move all funds from {api_keys.bet_from_address} to {to_address}."
    )

    starting_balance_of_from_eoa = get_balances(api_keys.public_key)
    starting_balance_of_from = get_balances(api_keys.bet_from_address)
    starting_balance_of_to_address = get_balances(to_address)

    if starting_balance_of_from_eoa.xdai < xDai(0.01):
        logger.error(f"We need at least some funds in xDai to pay for the fees.")
        return

    failed_steps: list[str] = []

    # Sell all active bets on Omen.
    try:
        sell_all(api_keys, 0, auto_withdraw=True)
    except Exception as e:
        failed_steps.append("sell_all")
        logger.error(f"Failed to sell all bets on Omen: {e}")

    # Redeem any positions on Omen.
    try:
        redeem_from_all_user_positions(api_keys)
    except Exception as e:
        failed_steps.append("redeem_from_all_user_positions")
        logger.error(f"Failed to redeem from all user positions on Omen: {e}")

    # Claim any bonds on Reality.
    try:
        claim_all_bonds_on_reality(api_keys)
    except Exception as e:
        failed_steps.append("claim_all_bonds_on_reality")
        logger.error(f"Failed to claim all bonds on Reality: {e}")

    # Keep a little of xDai to pay for the rest of transactions, deposit rest into wxDai.
    if starting_balance_of_from.xdai > xDai(0.1):
        try:
            WrappedxDaiContract().deposit(
                api_keys, (starting_balance_of_from.xdai - xDai(0.1)).as_xdai_wei.as_wei
            )
        except Exception as e:
            failed_steps.append("deposit")
            logger.error(f"Failed to deposit xDai into wxDai: {e}")

    # Transfer all tokens we work with to the new address.
    for token_contract in SAFE_COLLATERAL_TOKENS:
        try:
            token_contract.transferFrom(
                api_keys,
                api_keys.bet_from_address,
                to_address,
                token_contract.balanceOf(api_keys.bet_from_address),
            )
            logger.info(
                f"Transferred {token_contract.symbol()} from {api_keys.bet_from_address} to {to_address}."
            )
        except Exception as e:
            failed_steps.append("transferFrom")
            logger.error(f"Failed to transfer {token_contract.symbol()}: {e}")

    # Show the ending balances.
    time.sleep(1)  # Othwerise we would get stale data here.
    ending_balance_of_to_address = get_balances(to_address)
    logger.warning(f"Failed steps: {failed_steps}")
    logger.info(
        f"Transfered total of {ending_balance_of_to_address.total - starting_balance_of_to_address.total} to {to_address}."
    )
    logger.info(
        "You might want to run this script again in the future, in case there were any locked resources."
    )


if __name__ == "__main__":
    typer.run(main)
