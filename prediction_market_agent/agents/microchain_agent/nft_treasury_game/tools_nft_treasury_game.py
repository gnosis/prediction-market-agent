from enum import Enum

from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.gtypes import xdai_type
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.datetime_utc import DatetimeUTC
from prediction_market_agent_tooling.tools.utils import utcnow
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    TREASURY_THRESHOLD_BALANCE_TO_END_GAME,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.contracts import (
    AgentRegisterContract,
    SimpleTreasuryContract,
)
from prediction_market_agent.db.agent_communication import (
    fetch_count_unprocessed_transactions,
    pop_message,
)


class NFTGameStatus(str, Enum):
    on = "on"
    finished = "finished"


def get_nft_game_status(web3: Web3 | None = None) -> NFTGameStatus:
    treasury_balance = SimpleTreasuryContract().balances(web3=web3)

    if treasury_balance.total < TREASURY_THRESHOLD_BALANCE_TO_END_GAME:
        return NFTGameStatus.finished

    if get_end_datetime_of_current_round() < utcnow():
        return NFTGameStatus.finished

    return NFTGameStatus.on


def get_nft_game_is_finished(web3: Web3 | None = None) -> bool:
    return get_nft_game_status(web3=web3) == NFTGameStatus.finished


def get_end_datetime_of_previous_round() -> DatetimeUTC | None:
    # TODO: Dynamically from somewhere.
    return None


def get_end_datetime_of_current_round() -> DatetimeUTC:
    # TODO: Dynamically from somewhere.
    return DatetimeUTC(year=2025, month=3, day=6, hour=23, minute=59, second=59)


def get_start_datetime_of_next_round() -> DatetimeUTC | None:
    # TODO: Dynamically from somewhere.
    return None


def purge_all_messages(keys: APIKeys) -> None:
    register = AgentRegisterContract()
    logger.info(f"Purging messages for {keys.bet_from_address}.")

    popped = 0
    with register.with_registered_agent(api_keys=keys):
        while fetch_count_unprocessed_transactions(
            consumer_address=keys.bet_from_address
        ):
            pop_message(minimum_fee=xdai_type(0), api_keys=keys)
            popped += 1
            logger.info(f"Popped {popped} messages.")
