from prediction_market_agent_tooling.tools.utils import DatetimeUTC
from sqlmodel import col

from prediction_market_agent.db.models import ReplicatedMarket
from prediction_market_agent.db.sql_handler import SQLHandler


class ReplicatedMarketsTableHandler:
    def __init__(
        self,
        sqlalchemy_db_url: str | None = None,
    ):
        self.sql_handler = SQLHandler(
            model=ReplicatedMarket, sqlalchemy_db_url=sqlalchemy_db_url
        )

    def save_replicated_markets(self, markets: list[ReplicatedMarket]) -> None:
        """Save item to storage."""
        self.sql_handler.save_multiple(markets)

    def get_all(self) -> list[ReplicatedMarket]:
        return list(self.sql_handler.get_all())

    def get_recent(self, from_: DatetimeUTC) -> list[ReplicatedMarket]:
        return self.sql_handler.get_with_filter_and_order(
            query_filters=(col(ReplicatedMarket.created_at) >= from_,),
            order_by_column_name="created_at",
            order_desc=False,
        )
