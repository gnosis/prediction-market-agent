"""
Run this script on an Anvil instance to reset the initial configurations for the NFT game.
Usage:
    python run_reset_game.py <RPC_URL> <OUTPUT_DIR> <xDai_balance_per_agent> <new_balance_treasury_xdai>
"""

import typer
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.utils import check_not_none, utcnow

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    STARTING_AGENT_BALANCE,
    STARTING_TREASURY_BALANCE,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.contracts import (
    AgentCommunicationContract,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.game_history import (
    NFTGameRoundTableHandler,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.scripts.generate_report import (
    generate_report,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.scripts.reset_balance_anvil import (
    get_nft_game_is_finished_rpc_url,
    redistribute_nft_keys,
    reset_balances,
)

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
    now = utcnow()
    keys = APIKeys()

    if check_game_finished and not get_nft_game_is_finished_rpc_url(rpc_url=rpc_url):
        logger.info(f"Game not finished yet, exiting.")
        return

    game_round_table_handler = NFTGameRoundTableHandler()
    last_round = check_not_none(game_round_table_handler.get_previous_round())

    if do_report:
        generate_report(
            last_round=last_round,
            rpc_url=rpc_url,
            initial_xdai_balance_per_agent=STARTING_AGENT_BALANCE,
        )

    # Restart reset the keys and balances if game is to happen again.
    current_round = game_round_table_handler.get_current_round()
    if force_restart or (current_round is not None and not current_round.restarted):
        if purge_messages:
            logger.info("Purging all messages from the AgentCommunicationContract.")
            AgentCommunicationContract().purge_all_messages(keys)

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
            AgentCommunicationContract().purge_all_messages(keys)

        if current_round is not None:
            game_round_table_handler.set_as_restarted(current_round)


if __name__ == "__main__":
    APP()
