import pytest

from prediction_market_agent.tools.mech.utils import (
    MechResult,
    MechTool,
    mech_request_local,
)
from tests.utils import RUN_PAID_TESTS


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
@pytest.mark.parametrize("mech_tool", list(MechTool))
def test_mech_local(mech_tool: MechTool) -> None:
    result: MechResult = mech_request_local(
        question="Is the sky blue?",
        mech_tool=mech_tool,
    )
    assert 0 <= result.p_yes <= 1
