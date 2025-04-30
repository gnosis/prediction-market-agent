from typing import Sequence
from unittest.mock import patch

import pytest
from eth_account import Account
from eth_account.signers.local import LocalAccount
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.gtypes import ChecksumAddress, HexBytes
from prediction_market_agent_tooling.tools.safe import create_safe
from safe_eth.eth import EthereumClient
from safe_eth.safe.safe import SafeV141

from prediction_market_agent.agents.safe_guard_agent.safe_utils import (
    post_or_execute,
    sign_or_execute,
)


def create_test_safe(
    threshold: int,
    local_ethereum_client: EthereumClient,
    test_keys: APIKeys,
    custom_owners: Sequence[ChecksumAddress] = tuple(),
) -> tuple[SafeV141, list[LocalAccount]]:
    account = test_keys.get_account()
    random_owners = [Account.create() for _ in range(5)]
    safe_address = create_safe(
        ethereum_client=local_ethereum_client,
        account=account,
        owners=[account.address, *[o.address for o in random_owners], *custom_owners],
        salt_nonce=42,
        threshold=threshold,
    )
    assert safe_address is not None, "Safe needs to be deployed."
    deployed_safe = SafeV141(safe_address, local_ethereum_client)
    return deployed_safe, random_owners


@pytest.mark.parametrize(
    "threshold, use_owner_safe",
    [
        (1, False),
        (2, False),
        (3, False),
        (1, True),
        (2, True),
        (3, True),
    ],
)
def test_post_or_execute(
    threshold: int,
    use_owner_safe: bool,
    local_ethereum_client: EthereumClient,
    test_keys: APIKeys,
) -> None:
    owner_safe, _ = create_test_safe(1, local_ethereum_client, test_keys)
    test_keys = test_keys.model_copy(
        update={"SAFE_ADDRESS": owner_safe.address if use_owner_safe else None}
    )
    main_safe, _ = create_test_safe(
        threshold, local_ethereum_client, test_keys, custom_owners=[owner_safe.address]
    )
    tx = main_safe.build_multisig_tx(
        to=test_keys.bet_from_address, value=0, data=HexBytes("0x")
    )
    with patch(
        "prediction_market_agent.agents.safe_guard_agent.safe_utils.TransactionServiceApi.post_transaction"
    ) as mock_post_transaction:
        try:
            result = post_or_execute(main_safe, tx, test_keys)
            exp: Exception | None = None
        except Exception as e:
            result = None
            exp = e

        if threshold == 1:
            # In the case of single signed Safe, transaction should get executed right away.
            assert exp is None
            assert isinstance(
                result, dict
            ), "Should get executed, because threshold is met."
            mock_post_transaction.assert_not_called()
        else:
            # In case of more signers needed, but we aren't using owner safe, it should get posted to Safe API.
            assert exp is None
            assert result is None, "Should not get executed, because threshold is >1."
            mock_post_transaction.assert_called_once_with(tx)


@pytest.mark.parametrize(
    "threshold, use_owner_safe",
    [
        (1, False),
        (2, False),
        (3, False),
        (1, True),
        (2, True),
        (3, True),
    ],
)
def test_sign_or_execute(
    threshold: int,
    use_owner_safe: bool,
    local_ethereum_client: EthereumClient,
    test_keys: APIKeys,
) -> None:
    owner_safe, _ = create_test_safe(1, local_ethereum_client, test_keys)
    test_keys = test_keys.model_copy(
        update={"SAFE_ADDRESS": owner_safe.address if use_owner_safe else None}
    )
    main_safe, main_safe_owners = create_test_safe(
        threshold, local_ethereum_client, test_keys, custom_owners=[owner_safe.address]
    )
    tx = main_safe.build_multisig_tx(
        to=test_keys.bet_from_address, value=0, data=HexBytes("0x")
    )
    # Add one standard EOA signature right away. In contrast to `post_or_execute` test,
    # here we simulate that someone created and signed the transaction already.
    tx.sign(main_safe_owners[0].key)
    with patch(
        "prediction_market_agent.agents.safe_guard_agent.safe_utils.TransactionServiceApi.post_signatures"
    ) as mock_post_signatures:
        try:
            result = sign_or_execute(main_safe, tx, test_keys)
            exp: Exception | None = None
        except Exception as e:
            result = None
            exp = e

        if threshold <= 2:
            # In the case of enough of signers Safe, transaction should get executed right away.
            assert exp is None
            assert isinstance(
                result, dict
            ), "Should get executed, because threshold is met."
            mock_post_signatures.assert_not_called()
        else:
            # In case of more signers needed, but we aren't using owner safe, it should get posted to Safe API.
            assert exp is None
            assert result is None, "Should not get executed, because threshold is >2."
            mock_post_signatures.assert_called_once_with(tx.safe_tx_hash, tx.signatures)
