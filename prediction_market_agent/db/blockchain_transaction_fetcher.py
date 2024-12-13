from typing import Any

import polars as pl
import spice
from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from prediction_market_agent_tooling.tools.web3_utils import xdai_to_wei
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.microchain_agent_keys import (
    MicrochainAgentKeys,
)
from prediction_market_agent.db.blockchain_message_table_handler import (
    BlockchainMessageTableHandler,
)
from prediction_market_agent.db.models import BlockchainMessage
from prediction_market_agent.tools.message_utils import decompress_message
from prediction_market_agent.utils import APIKeys


class BlockchainTransactionFetcher:
    def __init__(self) -> None:
        self.blockchain_table_handler = BlockchainMessageTableHandler()

    def unzip_message_else_do_nothing(self, data_field: str) -> str:
        """We try decompressing the message, else we return the original data field."""
        try:
            return decompress_message(HexBytes(data_field))
        except:
            return data_field

    def blockchain_message_from_dune_df_row(
        self, consumer_address: ChecksumAddress, x: dict[str, Any]
    ) -> BlockchainMessage:
        return BlockchainMessage(
            consumer_address=consumer_address,
            transaction_hash=x["hash"],
            value_wei=str(x["value"]),
            block=str(x["block_number"]),
            sender_address=x["from"],
            data_field=self.unzip_message_else_do_nothing(x["data"]),
        )

    def fetch_unseen_transactions(
        self, consumer_address: ChecksumAddress
    ) -> list[BlockchainMessage]:
        keys = APIKeys()
        latest_blockchain_message = (
            self.blockchain_table_handler.fetch_latest_blockchain_message(
                consumer_address
            )
        )
        min_block_number = (
            0 if not latest_blockchain_message else latest_blockchain_message.block
        )
        # We order by block_time because it's used as partition on Dune.
        # We use >= for block because we might have lost transactions from the same block.
        # Additionally, processed tx_hashes are filtered out anyways.
        query = f'select * from gnosis.transactions where "to" = {Web3.to_checksum_address(consumer_address)} AND block_number >= {min_block_number} and value >= {xdai_to_wei(MicrochainAgentKeys().RECEIVER_MINIMUM_AMOUNT)} order by block_time asc'
        df = spice.query(query, api_key=keys.dune_api_key.get_secret_value())

        existing_hashes = self.blockchain_table_handler.fetch_all_transaction_hashes(
            consumer_address=consumer_address
        )
        # Filter out existing hashes - hashes are by default lowercase
        df = df.filter(~pl.col("hash").is_in([i.hex() for i in existing_hashes]))
        return [
            self.blockchain_message_from_dune_df_row(consumer_address, x).model_copy(
                deep=True  # To prevent `is not bound to a Session` error.
            )
            for x in df.iter_rows(named=True)
        ]

    def fetch_count_unprocessed_transactions(
        self, consumer_address: ChecksumAddress
    ) -> int:
        transactions = self.fetch_unseen_transactions(consumer_address=consumer_address)
        return len(transactions)

    def fetch_one_unprocessed_blockchain_message_and_store_as_processed(
        self, consumer_address: ChecksumAddress
    ) -> BlockchainMessage | None:
        """
        Method for fetching oldest unprocessed transaction sent to the consumer address.
        After being fetched, it is stored in the DB as processed.
        """
        transactions = self.fetch_unseen_transactions(consumer_address=consumer_address)
        if not transactions:
            return None

        # We only want the oldest non-processed message.
        blockchain_message = transactions[0]

        # mark unseen transaction as processed in DB
        self.blockchain_table_handler.save_multiple([blockchain_message])
        return blockchain_message
