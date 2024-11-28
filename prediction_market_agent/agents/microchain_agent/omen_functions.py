from microchain import Function
from prediction_market_agent_tooling.gtypes import xdai_type
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.data_models import (
    OMEN_BINARY_MARKET_OUTCOMES,
    TEST_CATEGORY,
)
from prediction_market_agent_tooling.markets.omen.omen import (
    omen_create_market_tx,
    redeem_from_all_user_positions,
)
from prediction_market_agent_tooling.tools.datetime_utc import DatetimeUTC

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
        keys = APIKeys()
        prev_balance = get_balance(keys, market_type=MarketType.OMEN)
        redeem_from_all_user_positions(keys)
        new_balance = get_balance(keys, market_type=MarketType.OMEN)
        currency = new_balance.currency.value
        if redeemed_amount := new_balance.amount - prev_balance.amount > 0:
            return (
                f"Redeemed {redeemed_amount} {currency} in winnings. New "
                f"balance: {new_balance.amount}{currency}."
            )
        return (
            f"No winnings to redeem. Balance remains: {new_balance.amount}{currency}."
        )


class CreatePredictionMarket(Function):
    # Hard-coded low value before it's tested out more and real use-case is required.
    INITIAL_FUNDS = xdai_type(0.01)

    @property
    def description(self) -> str:
        return f"Use this function to create a new prediction market on Omen. Question of the prediciton market can only be binary, in the Yes/No format. Using this function will cost you {CreatePredictionMarket.INITIAL_FUNDS} xDai."

    @property
    def example_args(self) -> list[str]:
        return ["Binary yes/no question", "Closing time"]

    def __call__(self, question: str, closing_time: str) -> str:
        keys = APIKeys()
        closing_time_date = DatetimeUTC.to_datetime_utc(closing_time)
        created_market = omen_create_market_tx(
            keys,
            initial_funds=CreatePredictionMarket.INITIAL_FUNDS,
            question=question,
            closing_time=closing_time_date,
            category=TEST_CATEGORY,  # Force test category to not show these markets on Presagio until we know it works fine.
            outcomes=OMEN_BINARY_MARKET_OUTCOMES,
            language="en",
            auto_deposit=True,
        )
        return f"Created prediction market with id {created_market.market_event.fixed_product_market_maker_checksummed} at url {created_market.url}."


# Functions that interact exclusively with Omen prediction markets
OMEN_FUNCTIONS: list[type[Function]] = [
    RedeemWinningBets,
    CreatePredictionMarket,
]
