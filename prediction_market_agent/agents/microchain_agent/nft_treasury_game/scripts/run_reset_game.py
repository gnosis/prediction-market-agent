"""
Run this script on an Anvil instance to reset the initial configurations for the NFT game.
Usage:
    python run_reset_game.py <RPC_URL> <OUTPUT_DIR> <xDai_balance_per_agent> <new_balance_treasury_xdai>
"""

import typer
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.utils import check_not_none

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    STARTING_AGENT_BALANCE,
    STARTING_TREASURY_BALANCE,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.game_history import (
    NFTGameRoundTableHandler,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.scripts.generate_report import (
    generate_report,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.scripts.reset_balance_anvil import (
    get_nft_game_is_finished_rpc_url,
    purge_all_messages,
    redistribute_nft_keys,
    reset_balances,
)
from prediction_market_agent.db.report_table_handler import ReportNFTGameTableHandler

APP = typer.Typer(pretty_exceptions_enable=False)


@APP.command()
def main(
    rpc_url: str,
    check_game_finished: bool = True,
    do_report: bool = True,
    set_balances: bool = True,
    redistribute_keys: bool = True,
    purge_messages: bool = True,
    force_restart: bool = False,
) -> None:
    keys = APIKeys()

    if check_game_finished and not get_nft_game_is_finished_rpc_url(rpc_url=rpc_url):
        logger.info(f"Game not finished yet, exiting.")
        return

    report_table_handler = ReportNFTGameTableHandler()
    game_round_table_handler = NFTGameRoundTableHandler()

    last_round = game_round_table_handler.get_previous_round()
    if (
        do_report
        and last_round is not None
        and last_round.started
        and not report_table_handler.get_reports_by_game_round_id(
            check_not_none(last_round.id)
        )
    ):
        logger.info(f"Generating the report for the game {last_round}")
        generate_report(
            last_round=last_round,
            rpc_url=rpc_url,
            initial_xdai_balance_per_agent=STARTING_AGENT_BALANCE,
        )
    else:
        logger.info(f"Not generating report, as no last round was found.")

    # Restart reset the keys and balances if game is to happen again.
    current_round = game_round_table_handler.get_current_round()
    if force_restart or (current_round is not None and not current_round.started):
        logger.info(f"Setting the game state for {current_round}.")
        if set_balances:
            reset_balances(
                rpc_url=rpc_url,
                new_balance_agents_xdai=STARTING_AGENT_BALANCE,
                new_balance_treasury_xdai=STARTING_TREASURY_BALANCE,
            )

        if redistribute_keys:
            redistribute_nft_keys(rpc_url=rpc_url)

        if purge_messages:
            logger.info("Purging all messages from the AgentCommunicationContract.")
            purge_all_messages(rpc_url=rpc_url, keys=keys)

        if current_round is not None:
            game_round_table_handler.set_as_started(current_round)
    else:
        logger.info(f"No game to set state for.")


if __name__ == "__main__":
    APP()
