import typing
from typing import Generator
from unittest.mock import PropertyMock, patch

import polars as pl
import pytest
from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.gtypes import xdai_type
from prediction_market_agent_tooling.tools.web3_utils import xdai_to_wei
from pydantic import SecretStr
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.messages_functions import (
    ReceiveMessage,
)
from prediction_market_agent.agents.microchain_agent.utils import compress_message
from prediction_market_agent.db.blockchain_transaction_fetcher import (
    BlockchainTransactionFetcher,
)
from prediction_market_agent.utils import DBKeys


@pytest.fixture(scope="module")
def agent2_address() -> ChecksumAddress:
    return Web3.to_checksum_address("0xb4D8C8BedE2E49b08d2A22485f72fA516116FE7F")


# Random transactions found on Gnosisscan.
MOCK_HASH_1 = "0x5ba6dd51d3660f98f02683e032daa35644d3f7f975975da3c2628a5b4b1f5cb6"
MOCK_HASH_2 = "0x429f61ea3e1afdd104fdd0a6f3b88432ec4c7b298fd126378e53a63bc60fed6a"


def mock_spice_query(query: str, api_key: str) -> pl.DataFrame:
    anvil_account_1 = Web3.to_checksum_address(
        "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
    )
    return pl.DataFrame(
        {
            "hash": [MOCK_HASH_1, MOCK_HASH_2],
            "value": [xdai_to_wei(xdai_type(1)), xdai_to_wei(xdai_type(2))],
            "block_number": [1, 2],
            "from": [anvil_account_1, anvil_account_1],
            "data": ["test", Web3.to_hex(compress_message("test"))],
        }
    )


@pytest.fixture(scope="module")
def patch_dune_api_key() -> Generator[PropertyMock, None, None]:
    with patch(
        "prediction_market_agent.utils.APIKeys.dune_api_key",
        new_callable=PropertyMock,
    ) as mock_dune:
        mock_dune.return_value = SecretStr("mock_dune_api_key")
        yield mock_dune


@pytest.fixture(scope="module")
def patch_spice() -> Generator[PropertyMock, None, None]:
    with patch(
        "spice.query",
        side_effect=mock_spice_query,
    ) as mock_spice:
        yield mock_spice


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
    session_keys_with_mocked_db: DBKeys,
) -> Generator[PropertyMock, None, None]:
    with patch(
        "prediction_market_agent_tooling.config.APIKeys.sqlalchemy_db_url",
        new_callable=PropertyMock,
    ) as mock_sqlalchemy_db_url:
        mock_sqlalchemy_db_url.return_value = (
            session_keys_with_mocked_db.SQLALCHEMY_DB_URL
        )
        yield mock_sqlalchemy_db_url


def test_receive_message_description(
    patch_pytest_db: PropertyMock,
    patch_public_key: PropertyMock,
    patch_spice: PropertyMock,
    patch_dune_api_key: PropertyMock,
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
    patch_pytest_db: PropertyMock,
    patch_public_key: PropertyMock,
    patch_spice: PropertyMock,
    patch_dune_api_key: PropertyMock,
) -> None:
    r = ReceiveMessage()

    blockchain_message = r()
    assert blockchain_message is not None
    assert blockchain_message.transaction_hash == MOCK_HASH_1


def test_receive_message_then_check_count_unseen_messages(
    patch_pytest_db: PropertyMock,
    patch_public_key: PropertyMock,
    patch_spice: typing.Any,
    patch_dune_api_key: PropertyMock,
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
