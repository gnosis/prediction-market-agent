from unittest.mock import PropertyMock, patch

from web3 import Web3

from prediction_market_agent.agents.safe_guard_agent.guards.agent import (
    validate_do_not_remove_agent,
)
from prediction_market_agent.agents.safe_guard_agent.safe_api_utils import (
    get_safe_detailed_transaction_info,
    safe_tx_from_detailed_transaction,
)
from prediction_market_agent.agents.safe_guard_agent.safe_utils import get_safe


def test_validate_do_not_remove_agent_accepts_arbitrary_tx() -> None:
    safe = get_safe(
        Web3.to_checksum_address("0x8f72555D97ad3364e65d64AC5aE4103C22d3e754")
    )
    detailed_tx = get_safe_detailed_transaction_info(
        "multisig_0x8f72555D97ad3364e65d64AC5aE4103C22d3e754_0x2b6e85fb8b4c0e717f5783fc117d7e1804eca118311021c058cc3a5e6d871aca"
    )
    safe_tx = safe_tx_from_detailed_transaction(safe, detailed_tx)
    assert validate_do_not_remove_agent(detailed_tx, safe_tx, [], []).ok


def test_validate_do_not_remove_agent_accepts_removal_of_others() -> None:
    safe = get_safe(
        Web3.to_checksum_address("0x8f72555D97ad3364e65d64AC5aE4103C22d3e754")
    )
    detailed_tx = get_safe_detailed_transaction_info(
        "multisig_0x8f72555D97ad3364e65d64AC5aE4103C22d3e754_0x8e42099a1bc122fb4496b0aa662201c87f45f2d5faa11a837456df051fc711ad"
    )
    safe_tx = safe_tx_from_detailed_transaction(safe, detailed_tx)
    assert validate_do_not_remove_agent(detailed_tx, safe_tx, [], []).ok


def test_validate_do_not_remove_agent_forbids_removal_of_agent_itself() -> None:
    with patch(
        "prediction_market_agent.agents.safe_guard_agent.guards.agent.APIKeys.bet_from_address",
        new_callable=PropertyMock,
    ) as mock_bet_from_address:
        # Simulate that APIKeys holds the address that is being removed in that transaction.
        mock_bet_from_address.return_value = (
            "0x8bc8a41B500fB012a36a5D46547f8dEaf8D5A1Fd"
        )
        safe = get_safe(
            Web3.to_checksum_address("0x8f72555D97ad3364e65d64AC5aE4103C22d3e754")
        )
        detailed_tx = get_safe_detailed_transaction_info(
            "multisig_0x8f72555D97ad3364e65d64AC5aE4103C22d3e754_0x8e42099a1bc122fb4496b0aa662201c87f45f2d5faa11a837456df051fc711ad"
        )
        safe_tx = safe_tx_from_detailed_transaction(safe, detailed_tx)
        assert not validate_do_not_remove_agent(detailed_tx, safe_tx, [], []).ok
