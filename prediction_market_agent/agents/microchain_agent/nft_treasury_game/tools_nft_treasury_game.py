from enum import Enum

from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.gtypes import HexBytes, xDai
from prediction_market_agent_tooling.loggers import logger
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    TREASURY_THRESHOLD_BALANCE_TO_END_GAME,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.contracts import (
    AgentRegisterContract,
    SimpleTreasuryContract,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.game_history import (
    NFTGameRoundTableHandler,
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

    if treasury_balance.xdai < TREASURY_THRESHOLD_BALANCE_TO_END_GAME:
        return NFTGameStatus.finished

    if not NFTGameRoundTableHandler().get_game_round_is_active():
        return NFTGameStatus.finished

    return NFTGameStatus.on


def get_nft_game_is_finished(web3: Web3 | None = None) -> bool:
    return get_nft_game_status(web3=web3) == NFTGameStatus.finished


def purge_all_messages(keys: APIKeys) -> None:
    register = AgentRegisterContract()
    logger.info(f"Purging messages for {keys.bet_from_address}.")

    popped = 0
    with register.with_registered_agent(api_keys=keys):
        while fetch_count_unprocessed_transactions(
            consumer_address=keys.bet_from_address
        ):
            pop_message(minimum_fee=xDai(0), api_keys=keys)
            popped += 1
            logger.info(f"Popped {popped} messages.")


def hash_password(password: str) -> str:
    return HexBytes(Web3.solidity_keccak(["string"], [password])).hex()
