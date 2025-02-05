"""
Run this script on an Anvil instance to reset the initial configurations for the NFT game.
Usage:
    python run_reset_game.py <RPC_URL> <OUTPUT_DIR> <xDai_balance_per_agent> <new_balance_treasury_xdai>
"""
from pathlib import Path
from typing import Annotated

import typer

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.scripts.generate_report import (
    generate_report,
)

APP = typer.Typer(pretty_exceptions_enable=False)


@APP.command()
def main(
    rpc_url: str,
    output_dir: Path,
    xdai_balance_per_agent: Annotated[int, typer.Argument()] = 200,
    new_balance_treasury_xdai: Annotated[int, typer.Argument()] = 100,
) -> None:
    # if not is_treasury_empty(rpc_url=rpc_url):
    #     logger.info(f"Treasury not empty, exiting.")
    #     return

    # Give time to agents to reflect on the last game, in case this script gets executed right after it ended
    # time.sleep(10 * 60)

    generate_report(
        rpc_url=rpc_url,
        initial_xdai_balance_per_agent=xdai_balance_per_agent,
    )
    # reset_balances(
    #     rpc_url=rpc_url,
    #     new_balance_agents_xdai=xdai_balance_per_agent,
    #     new_balance_treasury_xdai=new_balance_treasury_xdai,
    # )
    # redistribute_nft_keys(rpc_url=rpc_url)


if __name__ == "__main__":
    APP()
