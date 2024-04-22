from microchain import Function
from prediction_market_agent_tooling.markets.omen.omen import (
    redeem_from_all_user_positions,
)

from prediction_market_agent.agents.microchain_agent.utils import MicrochainAPIKeys


class RedeemWinningBets(Function):
    @property
    def description(self) -> str:
        return "Use this function to redeem winnings from a position that you opened which has already been resolved. Use this to retrieve funds from a bet you placed in a market, after the market has been resolved."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> None:
        redeem_from_all_user_positions(MicrochainAPIKeys().bet_from_private_key)


# Functions that interact exclusively with Omen prediction markets
OMEN_FUNCTIONS: list[type[Function]] = [RedeemWinningBets]
