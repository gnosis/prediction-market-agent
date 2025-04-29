import pytest
from web3 import Web3

from prediction_market_agent.agents.safe_guard_agent.safe_api_utils import (
    get_safe_queue,
)


def test_get_safe_queue_breaks() -> None:
    # When this test starts to fail (funny that means that Safe API is fixed),
    # we should be able to enable nested Safe support in SG agent.
    with pytest.raises(Exception):
        get_safe_queue(
            Web3.to_checksum_address("0xbDA90E5bA055972a2F757AC3F7f73740Ce60bE36")
        )
