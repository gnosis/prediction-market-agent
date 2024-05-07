from microchain import Function
from prediction_market_agent_tooling.config import PrivateCredentials
from prediction_market_agent_tooling.markets.omen.omen import \
    redeem_from_all_user_positions

from prediction_market_agent.utils import APIKeys


class RedeemWinningBets(Function):
    @property
    def description(self) -> str:
        return "Use this function to redeem winnings from a position that you opened which has already been resolved. Use this to retrieve funds from a bet you placed in a market, after the market has been resolved. If you have outstanding winnings to be redeemed, your balance will be updated."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> None:
        redeem_from_all_user_positions(PrivateCredentials.from_api_keys(APIKeys()))


# Functions that interact exclusively with Omen prediction markets
OMEN_FUNCTIONS: list[type[Function]] = [RedeemWinningBets]
