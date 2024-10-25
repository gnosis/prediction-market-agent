import pytest
from crewai import Task

from prediction_market_agent.agents.utils import get_maximum_possible_bet_amount
from prediction_market_agent.utils import disable_crewai_telemetry


def test_disable_crewai_telemetry() -> None:
    disable_crewai_telemetry()
    t = Task(
        description="foo",
        expected_output="bar",
    )
    assert not t._telemetry.task_started(task=t)


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
    assert get_maximum_possible_bet_amount(min_, max_, trading_balance) == expected
