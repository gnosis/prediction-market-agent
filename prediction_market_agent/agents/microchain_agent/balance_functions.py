from microchain import Function
from prediction_market_agent_tooling.tools.balances import get_balances
from web3 import Web3

from prediction_market_agent.utils import APIKeys


class GetOtherWalletXDAIBalance(Function):
    @property
    def description(self) -> str:
        return f"Use this function to fetch xDai and wxDai balance of someone else given his wallet address."

    @property
    def example_args(self) -> list[str]:
        return ["0xSomeAddress"]

    def __call__(self, wallet_address: str) -> str:
        balances = get_balances(Web3.to_checksum_address(wallet_address))
        return f"Balance of {wallet_address} is {balances.xdai} xDai and {balances.wxdai} wxDai and {balances.sdai} sDai."


class GetMyXDAIBalance(Function):
    @property
    def description(self) -> str:
        return f"Use this function to fetch your xDai and wxDai balance."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> str:
        balances = get_balances(APIKeys().bet_from_address)
        return f"You have {balances.xdai} xDai and {balances.wxdai} wxDai and {balances.sdai} sDai."


BALANCE_FUNCTIONS: list[type[Function]] = [GetOtherWalletXDAIBalance, GetMyXDAIBalance]
