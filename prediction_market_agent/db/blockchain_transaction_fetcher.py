import polars as pl
import spice
from eth_typing import ChecksumAddress
from prediction_market_agent_tooling.gtypes import xDai
from prediction_market_agent_tooling.tools.web3_utils import xdai_to_wei
from web3 import Web3

from prediction_market_agent.db.blockchain_message_table_handler import (
    BlockchainMessageTableHandler,
)
from prediction_market_agent.db.models import BlockchainMessage
from prediction_market_agent.utils import APIKeys


class BlockchainTransactionFetcher:
    MIN_TRANSACTION_AMOUNT = xDai(0.001)

    def __init__(self):
        self.blockchain_table_handler = BlockchainMessageTableHandler()

    def update_unprocessed_transactions_sent_to_address(
        self, consumer_address: ChecksumAddress
    ) -> list[BlockchainMessage]:
        """
        Method for fetching unprocessed transactions sent to the consumer address.
        After fetching these, they are stored in the DB as processed.
        """
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
        query = f'select * from gnosis.transactions where "from" = {Web3.to_checksum_address(consumer_address)} AND block_number >= {min_block_number} and value > {xdai_to_wei(self.MIN_TRANSACTION_AMOUNT)} order by block_time desc'
        df = spice.query(query=query, api_key=keys.dune_api_key.get_secret_value())
        # ToDo - Filter transactions whose hashes are already in the DB.
        existing_hashes = self.blockchain_table_handler.fetch_all_transaction_hashes(
            consumer_address=consumer_address
        )
        # Filter out existing hashes
        df = df.filter(~pl.col("hash").is_in(existing_hashes))
        # ToDo - Get latest hash and block_number, store
        # ToDo - After messages were processed, we store them in the DB.
        # ToDo - Return blockchain_objects, when processed, store in DB
        blockchain_messages = [
            BlockchainMessage(
                consumer_address=consumer_address,
                transaction_hash=row["hash"],
                value=row["value"],
                block=row["block_number"],
            )
            for row in df.iter_rows(named=True)
        ]
        # update DB
        self.blockchain_table_handler.save_multiple(blockchain_messages)
        return blockchain_messages
