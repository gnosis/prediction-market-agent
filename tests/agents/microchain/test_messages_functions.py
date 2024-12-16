import typing
from typing import Generator
from unittest.mock import PropertyMock, patch

import polars as pl
import pytest
from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from pydantic import SecretStr
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.messages_functions import (
    ReceiveMessage,
)
from prediction_market_agent.db.blockchain_message_table_handler import (
    BlockchainMessageTableHandler,
)
from prediction_market_agent.db.blockchain_transaction_fetcher import (
    BlockchainTransactionFetcher,
)
from prediction_market_agent.tools.message_utils import compress_message


@pytest.fixture(scope="session")
def account2_address() -> ChecksumAddress:
    # anvil account # 2
    return Web3.to_checksum_address("0x70997970C51812dc3A010C7d01b50e0d17dc79C8")


@pytest.fixture(scope="session")
def account2_private_key() -> SecretStr:
    "Anvil test account private key. It's public already."
    return SecretStr(
        "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
    )


# Random transactions found on Gnosisscan.
MOCK_HASH_1 = "0x5ba6dd51d3660f98f02683e032daa35644d3f7f975975da3c2628a5b4b1f5cb6"
MOCK_HASH_2 = "0x429f61ea3e1afdd104fdd0a6f3b88432ec4c7b298fd126378e53a63bc60fed6a"
MOCK_SENDER_SPICE_QUERY = Web3.to_checksum_address(
    "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
)  # anvil account 1


def mock_spice_query(query: str, api_key: str, cache: bool) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "hash": [MOCK_HASH_1, MOCK_HASH_2],
            "value": [Web3.to_wei(1, "ether"), Web3.to_wei(2, "ether")],
            "block_number": [1, 2],
            "from": [MOCK_SENDER_SPICE_QUERY, MOCK_SENDER_SPICE_QUERY],
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


@pytest.fixture(scope="module")
def patch_send_xdai() -> Generator[PropertyMock, None, None]:
    # Note that we patch where the function is called (see https://docs.python.org/3/library/unittest.mock.html#where-to-patch).
    with patch(
        "prediction_market_agent.agents.microchain_agent.messages_functions.send_xdai_to",
        return_value={"transactionHash": HexBytes(MOCK_HASH_1)},
    ) as mock_send_xdai:
        yield mock_send_xdai


@pytest.fixture
def patch_public_key(
    account2_address: ChecksumAddress, account2_private_key: SecretStr
) -> Generator[PropertyMock, None, None]:
    with patch(
        "prediction_market_agent.agents.microchain_agent.microchain_agent_keys.MicrochainAgentKeys.public_key",
        new_callable=PropertyMock,
    ) as mock_public_key, patch(
        "prediction_market_agent.agents.microchain_agent.microchain_agent_keys.MicrochainAgentKeys.bet_from_private_key",
        new_callable=PropertyMock,
    ) as mock_private_key:
        mock_public_key.return_value = account2_address
        mock_private_key.return_value = account2_private_key
        yield mock_public_key


@pytest.fixture(scope="function")
def patch_pytest_db(
    memory_blockchain_handler: BlockchainMessageTableHandler,
) -> Generator[PropertyMock, None, None]:
    with patch(
        "prediction_market_agent_tooling.config.APIKeys.sqlalchemy_db_url",
        new_callable=PropertyMock,
    ) as mock_sqlalchemy_db_url:
        mock_sqlalchemy_db_url.return_value = SecretStr("sqlite://")
        yield mock_sqlalchemy_db_url


def test_receive_message_description(
    patch_pytest_db: PropertyMock,
    patch_public_key: PropertyMock,
    patch_spice: PropertyMock,
    patch_dune_api_key: PropertyMock,
    patch_send_xdai: PropertyMock,
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
    patch_send_xdai: PropertyMock,
    patch_pytest_db: PropertyMock,
    patch_public_key: PropertyMock,
    patch_spice: PropertyMock,
    patch_dune_api_key: PropertyMock,
) -> None:
    r = ReceiveMessage()

    blockchain_message = r()
    assert blockchain_message is not None
    assert MOCK_SENDER_SPICE_QUERY in blockchain_message


def test_receive_message_then_check_count_unseen_messages(
    patch_pytest_db: PropertyMock,
    patch_public_key: PropertyMock,
    patch_spice: typing.Any,
    patch_dune_api_key: PropertyMock,
    patch_send_xdai: PropertyMock,
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
