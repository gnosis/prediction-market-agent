from enum import Enum

from prediction_market_agent_tooling.gtypes import xDai
from prediction_market_agent_tooling.tools.balances import get_balances

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    TREASURY_SAFE_ADDRESS,
)


class NFTGameStatus(str, Enum):
    on = "on"
    finished = "finished"


def get_nft_game_status() -> NFTGameStatus:
    treasury_balance = get_balances(TREASURY_SAFE_ADDRESS)

    if treasury_balance.total == xDai(0):
        return NFTGameStatus.finished

    return NFTGameStatus.on
