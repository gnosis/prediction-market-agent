import pytest
from prediction_market_agent_tooling.gtypes import USD

from prediction_market_agent.agents.utils import get_maximum_possible_bet_amount
from prediction_market_agent.tools.message_utils import (
    compress_message,
    decompress_message,
)


@pytest.mark.parametrize(
    "min_, max_, trading_balance, expected",
    [
        (1, 5, 3, 3 * 0.95),
        (1, 5, 100, 5),
        (1, 5, 0.1, 1),
    ],
)
def test_get_maximum_possible_bet_amount(
    min_: float, max_: float, trading_balance: float, expected: float
) -> None:
    assert get_maximum_possible_bet_amount(
        USD(min_), USD(max_), USD(trading_balance)
    ) == USD(expected)


def test_message_compression() -> None:
    message = "Hello!"
    encoded = compress_message(message)
    assert message == decompress_message(encoded)
