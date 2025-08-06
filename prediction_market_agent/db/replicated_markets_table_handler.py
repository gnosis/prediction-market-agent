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
