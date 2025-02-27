import time

import typer
from prediction_market_agent_tooling.gtypes import private_key_type, wei_type
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    SAFE_COLLATERAL_TOKENS,
)
from prediction_market_agent_tooling.tools.balances import get_balances
from prediction_market_agent_tooling.tools.tokens.auto_withdraw import (
    auto_withdraw_collateral_token,
)
from prediction_market_agent_tooling.tools.tokens.main_token import KEEPING_ERC20_TOKEN
from prediction_market_agent_tooling.tools.web3_utils import wei_to_xdai
from web3 import Web3

from prediction_market_agent.utils import APIKeys


def main(
    from_private_key: str,
    from_safe_address: str | None = None,
) -> None:
    """
    This script will transfer all erc 20 tokens that we usually work with into the KEEPING_ERC20_TOKEN.
    Use for example when agent scattered it funds through multiple tokens (wxDai, sDai, GNO, ...) and you want to consolidate them.
    If Safe is provided, it will transfer resources from there and to there, instead of the EOA.
    """
    api_keys = APIKeys(
        BET_FROM_PRIVATE_KEY=private_key_type(from_private_key),
        SAFE_ADDRESS=(
            Web3.to_checksum_address(from_safe_address) if from_safe_address else None
        ),
    )

    logger.info(
        f"Going to move all funds from {api_keys.bet_from_address} to its {KEEPING_ERC20_TOKEN.symbol_cached()}."
    )

    starting_balance_of_from_eoa = get_balances(api_keys.public_key)
    starting_balance_keeping_token = KEEPING_ERC20_TOKEN.balanceOf(
        api_keys.bet_from_address
    )

    if starting_balance_of_from_eoa.xdai < 0.01:
        logger.error(f"We need at least some funds in xDai to pay for the fees.")
        return

    failed_steps: list[str] = []

    # Transfer all tokens we work with to the new address.
    for token_contract in SAFE_COLLATERAL_TOKENS:
        try:
            auto_withdraw_collateral_token(
                token_contract,
                amount_wei=token_contract.balanceOf(api_keys.bet_from_address),
                api_keys=api_keys,
            )
        except Exception as e:
            failed_steps.append("transferFrom")
            logger.error(f"Failed to transfer {token_contract.symbol_cached()}: {e}")

    time.sleep(1)  # Othwerise we would get stale data here.
    ending_balance_keeping_token = KEEPING_ERC20_TOKEN.balanceOf(
        api_keys.bet_from_address
    )

    # Show the ending balances.
    logger.warning(f"Failed steps: {failed_steps}")
    logger.info(
        f"Transfered total of {wei_to_xdai(wei_type(ending_balance_keeping_token - starting_balance_keeping_token))} {KEEPING_ERC20_TOKEN.symbol_cached()}."
    )
    logger.info(
        "You might want to run this script again in the future, in case there were any locked resources."
    )


if __name__ == "__main__":
    typer.run(main)
