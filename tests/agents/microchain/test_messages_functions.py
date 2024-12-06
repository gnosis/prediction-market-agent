import os
from typing import Generator
from unittest.mock import PropertyMock, patch

import pytest
from eth_typing import ChecksumAddress
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.messages_functions import (
    ReceiveMessage,
)
from prediction_market_agent.db.blockchain_transaction_fetcher import (
    BlockchainTransactionFetcher,
)
from prediction_market_agent.utils import DBKeys


@pytest.fixture(scope="module")
def agent2_address() -> ChecksumAddress:
    return Web3.to_checksum_address("0xb4D8C8BedE2E49b08d2A22485f72fA516116FE7F")


@pytest.fixture
def patch_public_key(
    agent2_address: ChecksumAddress,
) -> Generator[PropertyMock, None, None]:
    with patch(
        "prediction_market_agent.agents.microchain_agent.microchain_agent_keys.MicrochainAgentKeys.public_key",
        new_callable=PropertyMock,
    ) as mock_public_key:
        mock_public_key.return_value = agent2_address
        yield mock_public_key


@pytest.fixture
def patch_pytest_db(
    session_keys_with_postgresql_proc_and_enabled_cache: DBKeys,
) -> Generator[PropertyMock, None, None]:
    # Mocking os.environ because mocking Pydantic's attribute is not possible via patch.
    with patch.dict(
        os.environ,
        {
            "SQLALCHEMY_DB_URL": session_keys_with_postgresql_proc_and_enabled_cache.SQLALCHEMY_DB_URL
        },
    ) as mock_db:
        yield mock_db


def test_receive_message_description(
    patch_pytest_db: PropertyMock, patch_public_key: PropertyMock
) -> None:
    r = ReceiveMessage()
    description = r.description
    count_unseen_messages = (
        BlockchainTransactionFetcher().fetch_count_unprocessed_transactions(
            patch_public_key.return_value
        )
    )
    assert str(count_unseen_messages) in description


def test_receive_message_call(
    patch_pytest_db: PropertyMock, patch_public_key: PropertyMock
) -> None:
    r = ReceiveMessage()
    # We expect at least 1 message since there was a test tx sent to agent 2.
    blockchain_message = r()
    assert blockchain_message is not None


def test_receive_message_then_check_count_unseen_messages(
    patch_pytest_db: PropertyMock, patch_public_key: PropertyMock
) -> None:
    # Idea here is to fetch the next message, and then fetch the count of unseen messages, asserting that
    # this number decreased by 1.
    r = ReceiveMessage()

    initial_count_unseen_messages = (
        BlockchainTransactionFetcher().fetch_count_unprocessed_transactions(
            patch_public_key.return_value
        )
    )

    r()
    current_count_unseen_messages = (
        BlockchainTransactionFetcher().fetch_count_unprocessed_transactions(
            patch_public_key.return_value
        )
    )
    assert current_count_unseen_messages == initial_count_unseen_messages - 1
