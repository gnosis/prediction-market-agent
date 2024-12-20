from prediction_market_agent_tooling.gtypes import wei_type
from web3 import Web3

from prediction_market_agent.db.blockchain_message_table_handler import (
    BlockchainMessageTableHandler,
)
from prediction_market_agent.db.models import BlockchainMessage

MOCK_ADDRESS = Web3.to_checksum_address(
    "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
)  # anvil 1


def test_save_blockchain_message(
    memory_blockchain_handler: BlockchainMessageTableHandler,
) -> None:
    mock_value_wei = wei_type(
        20000000000000000000
    )  # large enough value for surpassing bigint limitation on postgres | sqlite
    blockchain_message = BlockchainMessage(
        consumer_address=MOCK_ADDRESS,
        sender_address=MOCK_ADDRESS,
        block="123",
        transaction_hash="0x828f568506f24baca7314fda1430232d3c907520af1be714ba3b4f64e690555e",  # dummy
        value_wei=str(mock_value_wei),
        data_field="0x",
    )

    # assert DB is empty
    assert not memory_blockchain_handler.sql_handler.get_all()
    memory_blockchain_handler.save_multiple([blockchain_message])
    assert len(memory_blockchain_handler.sql_handler.get_all()) == 1
