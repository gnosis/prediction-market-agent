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


class GetAgentsInTheGame(Function):
    @property
    def description(self) -> str:
        return f"Returns a list of all agents that are currently registered in the game and can receive messages."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> str:
        register = AgentRegisterContract()
        addresses = register.get_all_registered_agents()
        return "Agents currently registered in the game are:\n" + "\n".join(addresses)


class LearnAboutNFTContractWithKeys(Function):
    @property
    def description(self) -> str:
        return "Returns information about the contract holding keys to the treasury."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> str:
        n_nft_keys = NFTKeysContract.retrieve_total_number_of_keys()
        return f"Address of the NFT contract is {NFTKeysContract().address}, there are {n_nft_keys} keys, with token_id {list(range(n_nft_keys))}."


class LearnAboutTreasuryContract(Function):
    @property
    def description(self) -> str:
        return "Returns information about the treasury contract."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> str:
        return f"Address of the treasury contract is {SimpleTreasuryContract().address}. You need at least {SimpleTreasuryContract().required_nft_balance()} NFT keys to withdraw from the treasury. Current balance is {SimpleTreasuryContract().balances().xdai} xDai."


class WithdrawFromTreasury(Function):
    @property
    def description(self) -> str:
        required_balance_nft_tokens = SimpleTreasuryContract().required_nft_balance()
        return f"Transfers the entire balance of the treasury to the caller. For the function to succeed, the caller must own {required_balance_nft_tokens} NFT tokens."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> str:
        keys = MicrochainAgentKeys()
        treasury_contract = SimpleTreasuryContract()
        logger.info(
            f"Withdrawing from the treasury using sender {keys.bet_from_address}"
        )
        treasury_contract.withdraw(api_keys=keys)
        return "Treasury successfully emptied."


NFT_GAME_FUNCTIONS: list[type[Function]] = [
    GetAgentsInTheGame,
    LearnAboutNFTContractWithKeys,
    LearnAboutTreasuryContract,
    WithdrawFromTreasury,
]
