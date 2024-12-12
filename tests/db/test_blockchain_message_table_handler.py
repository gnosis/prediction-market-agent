from typing import Generator

import pytest
from eth.constants import ZERO_ADDRESS
from web3 import Web3

from prediction_market_agent.db.blockchain_message_table_handler import (
    BlockchainMessageTableHandler,
)
from prediction_market_agent.db.models import BlockchainMessage
from tests.db.conftest import reset_init_params_db_manager

SQLITE_DB_URL = "sqlite://"


@pytest.fixture(scope="function")
def memory_blockchain_handler() -> Generator[BlockchainMessageTableHandler, None, None]:
    """Creates a in-memory SQLite DB for testing"""
    prompt_handler = BlockchainMessageTableHandler(
        sqlalchemy_db_url=SQLITE_DB_URL,
    )
    yield prompt_handler
    reset_init_params_db_manager(prompt_handler.sql_handler.db_manager)


def test_save_prompt(memory_blockchain_handler: BlockchainMessageTableHandler) -> None:
    mock_address = ZERO_ADDRESS
    mock_value_wei = 20000000000000000000
    blockchain_message = BlockchainMessage(
        consumer_address=str(mock_address),
        sender_address=str(mock_address),
        block=0,
        transaction_hash="0x0000000000000000000000000000000000000000000000000000000000000000",
        value_wei=mock_value_wei,
        data_field="0x",
    )
    memory_blockchain_handler.save_multiple([blockchain_message])
    result = memory_blockchain_handler.fetch_latest_blockchain_message(
        Web3.to_checksum_address(mock_address)
    )
    assert result
    assert result.value_wei == mock_value_wei
