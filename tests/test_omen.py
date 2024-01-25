import time
from prediction_market_agent.utils import get_keys
from prediction_market_agent.markets.all_markets import omen
from prediction_market_agent.tools.gtypes import xdai_type


def test_omen_pick_binary_market() -> None:
    market = omen.pick_binary_market()
    assert market.outcomes == [
        "Yes",
        "No",
    ], "Omen binary market should have two outcomes, Yes and No."


def test_omen_get_market() -> None:
    market = omen.get_market("0xa3e47bb771074b33f2e279b9801341e9e0c9c6d7")
    assert (
        market.question
        == "Will Bethesda's 'Indiana Jones and the Great Circle' be released by January 25, 2024?"
    ), "Omen market question doesn't match the expected value."


def test_omen_buy_and_sell_outcome() -> None:
    # Tests both buying and selling, so we are back at the square one in the wallet (minues fees).
    market = omen.pick_binary_market()
    amount = xdai_type(0.001)
    keys = get_keys()
    omen.binary_omen_buy_outcome_tx(
        amount=amount,
        from_address=keys.bet_from_address,
        from_private_key=keys.bet_from_private_key,
        market=market,
        binary_outcome=True,
        auto_deposit=True,
    )
    time.sleep(3.14)  # Wait for the transaction to be mined.
    omen.binary_omen_sell_outcome_tx(
        amount=amount,
        from_address=keys.bet_from_address,
        from_private_key=keys.bet_from_private_key,
        market=market,
        binary_outcome=True,
        auto_withdraw=True,
    )
