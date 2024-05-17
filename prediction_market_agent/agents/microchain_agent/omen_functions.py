from microchain import Function
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.omen import (
    redeem_from_all_user_positions,
)

from prediction_market_agent.agents.microchain_agent.utils import get_balance
from prediction_market_agent.utils import APIKeys


class RedeemWinningBets(Function):
    @property
    def description(self) -> str:
        return "Use this function to redeem winnings from a position that you opened which has already been resolved. Use this to retrieve funds from a bet you placed in a market, after the market has been resolved. If you have outstanding winnings to be redeemed, your balance will be updated."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> str:
        prev_balance = get_balance(market_type=MarketType.OMEN)
        redeem_from_all_user_positions(APIKeys())
        new_balance = get_balance(market_type=MarketType.OMEN)
        currency = new_balance.currency.value
        if redeemed_amount := new_balance.amount - prev_balance.amount > 0:
            return (
                f"Redeemed {redeemed_amount} {currency} in winnings. New "
                f"balance: {new_balance.amount}{currency}."
            )
        return (
            f"No winnings to redeem. Balance remains: {new_balance.amount}{currency}."
        )


# Functions that interact exclusively with Omen prediction markets
OMEN_FUNCTIONS: list[type[Function]] = [RedeemWinningBets]
