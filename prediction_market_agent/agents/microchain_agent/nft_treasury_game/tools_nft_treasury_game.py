from enum import Enum

from prediction_market_agent_tooling.tools.balances import get_balances
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    TREASURY_ADDRESS,
    TREASURY_THRESHOLD_BALANCE_TO_END_GAME,
)


class NFTGameStatus(str, Enum):
    on = "on"
    finished = "finished"


def get_nft_game_status(web3: Web3 | None = None) -> NFTGameStatus:
    treasury_balance = get_balances(TREASURY_ADDRESS, web3=web3)

    if treasury_balance.total < TREASURY_THRESHOLD_BALANCE_TO_END_GAME:
        return NFTGameStatus.finished

    return NFTGameStatus.on
