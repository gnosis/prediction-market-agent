import typing as t

from prediction_market_agent_tooling.gtypes import ChecksumAddress
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from sqlalchemy import ColumnElement
from sqlmodel import col

from prediction_market_agent.db.models import BlockchainMessage
from prediction_market_agent.db.sql_handler import SQLHandler


class BlockchainMessageTableHandler:
    def __init__(
        self,
        sqlalchemy_db_url: str | None = None,
    ):
        self.sql_handler = SQLHandler(
            model=BlockchainMessage, sqlalchemy_db_url=sqlalchemy_db_url
        )

    def __build_consumer_column_filter(
        self, consumer_address: ChecksumAddress
    ) -> ColumnElement[bool]:
        return col(BlockchainMessage.consumer_address) == consumer_address

    def fetch_latest_blockchain_message(
        self, consumer_address: ChecksumAddress
    ) -> BlockchainMessage | None:
        query_filter = self.__build_consumer_column_filter(consumer_address)
        items: t.Sequence[
            BlockchainMessage
        ] = self.sql_handler.get_with_filter_and_order(
            query_filters=[query_filter],
            order_by_column_name=BlockchainMessage.block.key,  # type: ignore[attr-defined]
            order_desc=True,
            limit=1,
        )
        return items[0] if items else None

    def fetch_all_transaction_hashes(
        self, consumer_address: ChecksumAddress
    ) -> list[HexBytes]:
        query_filter = self.__build_consumer_column_filter(consumer_address)
        items: t.Sequence[
            BlockchainMessage
        ] = self.sql_handler.get_with_filter_and_order(query_filters=[query_filter])
        tx_hashes = [HexBytes(i.transaction_hash) for i in items]
        return list(set(tx_hashes))

    def save_multiple(self, items: t.Sequence[BlockchainMessage]) -> None:
        return self.sql_handler.save_multiple(
            # Re-create the items to avoid SQLModel errors. This is a workaround. It's weird, but it works. :shrug:
            [BlockchainMessage(**i.model_dump()) for i in items]
        )
