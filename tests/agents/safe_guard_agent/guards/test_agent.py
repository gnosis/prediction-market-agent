from unittest.mock import PropertyMock, patch

from prediction_market_agent_tooling.chains import GNOSIS_CHAIN_ID
from web3 import Web3

from prediction_market_agent.agents.safe_watch_agent.safe_api_utils import (
    get_safe_detailed_transaction_info,
    safe_tx_from_detailed_transaction,
)
from prediction_market_agent.agents.safe_watch_agent.safe_utils import get_safe
from prediction_market_agent.agents.safe_watch_agent.watchers.agent import (
    DoNotRemoveAgent,
)


def test_validate_do_not_remove_agent_accepts_arbitrary_tx() -> None:
    chain_id = GNOSIS_CHAIN_ID
    safe = get_safe(
        Web3.to_checksum_address("0x8f72555D97ad3364e65d64AC5aE4103C22d3e754"), chain_id
    )
    detailed_tx = get_safe_detailed_transaction_info(
        "multisig_0x8f72555D97ad3364e65d64AC5aE4103C22d3e754_0x2b6e85fb8b4c0e717f5783fc117d7e1804eca118311021c058cc3a5e6d871aca",
        chain_id=chain_id,
    )
    safe_tx = safe_tx_from_detailed_transaction(safe, detailed_tx)
    assert DoNotRemoveAgent().validate(detailed_tx, safe_tx, [], [], chain_id).ok


def test_validate_do_not_remove_agent_accepts_removal_of_others() -> None:
    chain_id = GNOSIS_CHAIN_ID
    safe = get_safe(
        Web3.to_checksum_address("0x8f72555D97ad3364e65d64AC5aE4103C22d3e754"), chain_id
    )
    detailed_tx = get_safe_detailed_transaction_info(
        "multisig_0x8f72555D97ad3364e65d64AC5aE4103C22d3e754_0x8e42099a1bc122fb4496b0aa662201c87f45f2d5faa11a837456df051fc711ad",
        chain_id=chain_id,
    )
    safe_tx = safe_tx_from_detailed_transaction(safe, detailed_tx)
    assert DoNotRemoveAgent().validate(detailed_tx, safe_tx, [], [], chain_id).ok


def test_validate_do_not_remove_agent_forbids_removal_of_agent_itself() -> None:
    chain_id = GNOSIS_CHAIN_ID
    with patch(
        "prediction_market_agent.agents.safe_watch_agent.watchers.agent.APIKeys.bet_from_address",
        new_callable=PropertyMock,
    ) as mock_bet_from_address:
        # Simulate that APIKeys holds the address that is being removed in that transaction.
        mock_bet_from_address.return_value = (
            "0x8bc8a41B500fB012a36a5D46547f8dEaf8D5A1Fd"
        )
        safe = get_safe(
            Web3.to_checksum_address("0x8f72555D97ad3364e65d64AC5aE4103C22d3e754"),
            chain_id,
        )
        detailed_tx = get_safe_detailed_transaction_info(
            "multisig_0x8f72555D97ad3364e65d64AC5aE4103C22d3e754_0x8e42099a1bc122fb4496b0aa662201c87f45f2d5faa11a837456df051fc711ad",
            chain_id=chain_id,
        )
        safe_tx = safe_tx_from_detailed_transaction(safe, detailed_tx)
        assert (
            not DoNotRemoveAgent().validate(detailed_tx, safe_tx, [], [], chain_id).ok
        )
