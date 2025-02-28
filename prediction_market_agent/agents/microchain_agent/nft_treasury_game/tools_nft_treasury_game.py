from enum import Enum

from prediction_market_agent_tooling.tools.datetime_utc import DatetimeUTC
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    TREASURY_THRESHOLD_BALANCE_TO_END_GAME,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.contracts import (
    SimpleTreasuryContract,
)


class NFTGameStatus(str, Enum):
    on = "on"
    finished = "finished"


def get_nft_game_status(web3: Web3 | None = None) -> NFTGameStatus:
    treasury_balance = SimpleTreasuryContract().balances(web3=web3)

    if treasury_balance.total < TREASURY_THRESHOLD_BALANCE_TO_END_GAME:
        return NFTGameStatus.finished

    return NFTGameStatus.on


def get_nft_game_is_finished(web3: Web3 | None = None) -> bool:
    return get_nft_game_status(web3=web3) == NFTGameStatus.finished


def get_end_datetime_of_previous_round() -> DatetimeUTC:
    # TODO: Dynamically from somewhere.
    return DatetimeUTC(year=2025, month=2, day=28, hour=9)


def get_end_datetime_of_current_round() -> DatetimeUTC:
    # TODO: Dynamically from somewhere.
    return DatetimeUTC(year=2025, month=3, day=1, hour=9)


def get_start_datetime_of_next_round() -> DatetimeUTC:
    # TODO: Dynamically from somewhere.
    return DatetimeUTC(year=2025, month=3, day=3, hour=9)
