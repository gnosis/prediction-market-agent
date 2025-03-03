from microchain import Function
from prediction_market_agent_tooling.loggers import logger

from prediction_market_agent.agents.microchain_agent.microchain_agent_keys import (
    MicrochainAgentKeys,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.contracts import (
    AgentRegisterContract,
    NFTKeysContract,
    SimpleTreasuryContract,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.tools_nft_treasury_game import (
    get_end_datetime_of_current_round,
    get_nft_game_is_finished,
    get_start_datetime_of_next_round,
)


def get_game_has_ended_message() -> str:
    return f"The game round has ended, please check in later. Next round will start at {get_start_datetime_of_next_round()}."


class GetAgentsInTheGame(Function):
    @property
    def description(self) -> str:
        return f"Returns a list of all agents that are currently registered in the game and can receive messages."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> str:
        if get_nft_game_is_finished():
            return get_game_has_ended_message()
        register = AgentRegisterContract()
        addresses = register.get_all_registered_agents()
        return "Agents currently registered in the game are:\n" + "\n".join(addresses)


class WithdrawFromTreasury(Function):
    @property
    def description(self) -> str:
        required_balance_nft_tokens = SimpleTreasuryContract().required_nft_balance()
        return f"Transfers the entire balance of the treasury to the caller. For the function to succeed, the caller must own {required_balance_nft_tokens} NFT tokens."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> str:
        if get_nft_game_is_finished():
            return get_game_has_ended_message()
        keys = MicrochainAgentKeys()
        treasury_contract = SimpleTreasuryContract()
        logger.info(
            f"Withdrawing from the treasury using sender {keys.bet_from_address}"
        )
        treasury_contract.withdraw(api_keys=keys)
        return "Treasury successfully emptied."


class LearnAboutTheNFTGame(Function):
    @property
    def description(self) -> str:
        return "Returns all the information about the current state of the NFT game."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> str:
        if get_nft_game_is_finished():
            return get_game_has_ended_message()
        treasury = SimpleTreasuryContract()
        n_nft_keys = NFTKeysContract.retrieve_total_number_of_keys()
        return f"""Current state of the NFT Game:
        
Address of the NFT key contract is {NFTKeysContract().address}, there are {n_nft_keys} keys, with token_id {list(range(n_nft_keys))}."

Address of the treasury contract is {treasury.address}. You need at least {treasury.required_nft_balance()} NFT keys to withdraw from the treasury. Current balance is {treasury.balances().xdai} xDai.

If no one is able to withdraw from the treasury, the game will end on {get_end_datetime_of_current_round()} and the next round will start on {get_start_datetime_of_next_round()}.
"""


NFT_GAME_FUNCTIONS: list[type[Function]] = [
    GetAgentsInTheGame,
    LearnAboutTheNFTGame,
    WithdrawFromTreasury,
]
