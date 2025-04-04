from sqlmodel import col

from prediction_market_agent.db.models import ReportNFTGame
from prediction_market_agent.db.sql_handler import SQLHandler


class ReportNFTGameTableHandler:
    def __init__(
        self,
        sqlalchemy_db_url: str | None = None,
    ):
        self.sql_handler = SQLHandler(
            model=ReportNFTGame, sqlalchemy_db_url=sqlalchemy_db_url
        )

    def save_report(self, report: ReportNFTGame) -> None:
        """Save item to storage."""
        self.sql_handler.save_multiple([report])

    def get_reports_by_game_round_id(self, game_round_id: int) -> list[ReportNFTGame]:
        """Get reports by game round id."""
        return self.sql_handler.get_with_filter_and_order(
            query_filters=[col(ReportNFTGame.game_round_id) == game_round_id]
        )
