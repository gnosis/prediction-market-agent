import polars as pl
import spice
from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.gtypes import xDai
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from prediction_market_agent_tooling.tools.web3_utils import xdai_to_wei
from web3 import Web3

from prediction_market_agent.agents.microchain_agent.utils import decompress_message
from prediction_market_agent.db.blockchain_message_table_handler import (
    BlockchainMessageTableHandler,
)
from prediction_market_agent.db.models import BlockchainMessage
from prediction_market_agent.utils import APIKeys


class BlockchainTransactionFetcher:
    MIN_TRANSACTION_AMOUNT = xDai(0.001)

    def __init__(self):
        self.blockchain_table_handler = BlockchainMessageTableHandler()

    def unzip_message_else_do_nothing(self, data_field: str) -> str:
        """We try decompressing the message, else we return the original data field."""
        try:
            return decompress_message(HexBytes(data_field))
        except:
            return data_field

    def fetch_unseen_transactions_df(
        self, consumer_address: ChecksumAddress
    ) -> pl.DataFrame:
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
        query = f'select * from gnosis.transactions where "from" = {Web3.to_checksum_address(consumer_address)} AND block_number >= {min_block_number} and value > {xdai_to_wei(self.MIN_TRANSACTION_AMOUNT)} order by block_time asc'
        df = spice.query(query=query, api_key=keys.dune_api_key.get_secret_value())
        existing_hashes = self.blockchain_table_handler.fetch_all_transaction_hashes(
            consumer_address=consumer_address
        )
        # Filter out existing hashes
        df = df.filter(~pl.col("hash").is_in(existing_hashes))
        return df

    def fetch_count_unprocessed_transactions(
        self, consumer_address: ChecksumAddress
    ) -> int:
        df = self.fetch_unseen_transactions_df(consumer_address=consumer_address)
        return len(df)

    def fetch_one_unprocessed_transaction_sent_to_address_and_store_as_processed(
        self, consumer_address: ChecksumAddress
    ) -> BlockchainMessage | None:
        """
        Method for fetching oldest unprocessed transaction sent to the consumer address.
        After being fetched, it is stored in the DB as processed.
        """
        df = self.fetch_unseen_transactions_df(consumer_address=consumer_address)
        if df.is_empty():
            return None

        # We only want the oldest non-processed message.
        oldest_non_processed_message = df.row(0, named=True)
        blockchain_message = BlockchainMessage(
            consumer_address=consumer_address,
            transaction_hash=oldest_non_processed_message["hash"],
            value_wei=oldest_non_processed_message["value"],
            block=oldest_non_processed_message["block_number"],
            sender_address=oldest_non_processed_message["from"],
            data_field=self.unzip_message_else_do_nothing(
                oldest_non_processed_message["data"]
            ),
        )

        # mark unseen transaction as processed in DB
        self.blockchain_table_handler.save_multiple([blockchain_message])
        return blockchain_message
