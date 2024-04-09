import pytest

from prediction_market_agent.agents.microchain_agent.functions import (
    BuyNo,
    BuyYes,
    GetMarkets,
)
from prediction_market_agent.agents.microchain_agent.utils import (
    get_omen_binary_markets,
)
from tests.utils import RUN_PAID_TESTS


def test_get_markets() -> None:
    get_markets = GetMarkets()
    assert len(get_markets()) > 0


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
def test_buy_yes() -> None:
    market = get_omen_binary_markets()[0]
    buy_yes = BuyYes()
    print(buy_yes(market.question, 0.0001))


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
def test_buy_no() -> None:
    market = get_omen_binary_markets()[0]
    buy_yes = BuyNo()
    print(buy_yes(market.question, 0.0001))
