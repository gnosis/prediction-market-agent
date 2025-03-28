import typing as t

from microchain import Function
from prediction_market_agent_tooling.loggers import logger

from prediction_market_agent.agents.microchain_agent.microchain_agent_keys import (
    MicrochainAgentKeys,
)
from prediction_market_agent.agents.microchain_agent.nft_functions import OwnerOfNFT
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
from prediction_market_agent.db.report_table_handler import (
    ReportNFTGame,
    ReportNFTGameTableHandler,
)


def get_game_has_ended_message() -> str:
    message = f"The game round has ended, please check in later."

    if (start_of_next_round := get_start_datetime_of_next_round()) is not None:
        message += f" The next round will start at {start_of_next_round}."

    return message


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
        owned_nft_keys = NFTKeysContract().balanceOf(
            MicrochainAgentKeys().bet_from_address
        )
        owned_nft_keys_message = (
            "You currently don't own any NFT keys."
            if not owned_nft_keys
            else f"You currently own {owned_nft_keys} NFT keys. You can use tool `{OwnerOfNFT.__name__}` to learn which keys you own."
        )
        message = f"""Current state of the NFT Game:
        
Address of the NFT key contract is {NFTKeysContract().address}, there are {n_nft_keys} keys, with token_id {list(range(n_nft_keys))}."

Address of the treasury contract is {treasury.address}. You need at least {treasury.required_nft_balance()} NFT keys to withdraw from the treasury. 

Current balance of the treasury is {treasury.balances().xdai} xDai.

{owned_nft_keys_message}

If no one is able to withdraw from the treasury, the game will end on {get_end_datetime_of_current_round()}."""
        if (start_of_next_round := get_start_datetime_of_next_round()) is not None:
            message += f" The next round will start at {start_of_next_round}."
        return message


class GetReportAboutThePreviousRound(Function):
    @property
    def description(self) -> str:
        return "Returns a report about the previous round of the NFT game, if there was any. You can use this for example to learn about the learnings from the previous round."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> str:
        report_handler = ReportNFTGameTableHandler()
        all_reports: t.Sequence[ReportNFTGame] = report_handler.sql_handler.get_all()
        overall_reports = [report for report in all_reports if report.is_overall_report]
        overall_reports.sort(key=lambda r: r.datetime_, reverse=True)

        if not overall_reports:
            if not get_nft_game_is_finished():
                return "There is no report about the previous round of the NFT game, because this is the first round and it is still active. Please participate in the game!"

            else:
                return "There is no report about the previous round of the NFT game, please try again later."

        return f"""The report is from {overall_reports[0].datetime_}:

{overall_reports[0].learnings}.

---

Currently, the game is {'not active' if get_nft_game_is_finished() else 'active'} and you should participate!"""


NFT_GAME_FUNCTIONS: list[type[Function]] = [
    GetAgentsInTheGame,
    LearnAboutTheNFTGame,
    WithdrawFromTreasury,
    GetReportAboutThePreviousRound,
]
