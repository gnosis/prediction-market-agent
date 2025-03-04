"""
Run this script on an Anvil instance to reset the initial configurations for the NFT game.
Usage:
    python run_reset_game.py <RPC_URL> <OUTPUT_DIR> <xDai_balance_per_agent> <new_balance_treasury_xdai>
"""

import time

import typer
from prediction_market_agent_tooling.loggers import logger

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    STARTING_AGENT_BALANCE,
    STARTING_TREASURY_BALANCE,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.scripts.generate_report import (
    generate_report,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.scripts.reset_balance_anvil import (
    is_game_in_finished_state,
    redistribute_nft_keys,
    reset_balances,
)

APP = typer.Typer(pretty_exceptions_enable=False)


@APP.command()
def main(
    rpc_url: str,
    check_game_finished: bool = True,
    sleep: bool = True,
    do_report: bool = True,
    set_balances: bool = True,
) -> None:
    if check_game_finished and not is_game_in_finished_state(rpc_url=rpc_url):
        logger.info(f"Treasury not empty, exiting.")
        return

    # Give time to agents to reflect on the last game, in case this script gets executed right after it ended
    if sleep:
        time.sleep(10 * 60)

    if do_report:
        generate_report(
            rpc_url=rpc_url,
            initial_xdai_balance_per_agent=STARTING_AGENT_BALANCE,
        )
    if set_balances:
        reset_balances(
            rpc_url=rpc_url,
            new_balance_agents_xdai=STARTING_AGENT_BALANCE,
            new_balance_treasury_xdai=STARTING_TREASURY_BALANCE,
        )
    redistribute_nft_keys(rpc_url=rpc_url)


if __name__ == "__main__":
    APP()
