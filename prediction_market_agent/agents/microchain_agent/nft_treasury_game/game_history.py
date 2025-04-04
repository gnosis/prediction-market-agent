from typing import Optional

from prediction_market_agent_tooling.tools.datetime_utc import DatetimeUTC
from prediction_market_agent_tooling.tools.utils import utcnow
from sqlmodel import Field, SQLModel, col

from prediction_market_agent.db.sql_handler import SQLHandler


class NFTGameRound(SQLModel, table=True):
    __tablename__ = "nft_game_round"
    __table_args__ = {
        "extend_existing": True,
    }
    id: Optional[int] = Field(default=None, primary_key=True)
    start_time: DatetimeUTC
    end_time: DatetimeUTC
    started: bool = False
    created_at: DatetimeUTC = Field(default_factory=utcnow)


class NFTGameRoundTableHandler:
    def __init__(
        self,
        sqlalchemy_db_url: str | None = None,
    ):
        self.sql_handler = SQLHandler(
            model=NFTGameRound, sqlalchemy_db_url=sqlalchemy_db_url
        )

    def get_previous_round(self) -> NFTGameRound | None:
        prev_rounds: list[NFTGameRound] = self.sql_handler.get_with_filter_and_order(
            query_filters=[col(NFTGameRound.end_time) < utcnow()],
            limit=1,
            order_by_column_name=NFTGameRound.end_time.key,  # type: ignore
            order_desc=True,
        )
        return prev_rounds[0] if prev_rounds else None

    def get_current_round(self) -> NFTGameRound | None:
        now = utcnow()
        rounds: list[NFTGameRound] = self.sql_handler.get_with_filter_and_order(
            query_filters=[
                col(NFTGameRound.start_time) < now,
                col(NFTGameRound.end_time) > now,
            ],
        )
        if len(rounds) > 1:
            raise ValueError(
                f"More than one round is active at the same time: {rounds}"
            )
        return rounds[0] if rounds else None

    def get_next_round(self) -> NFTGameRound | None:
        next_rounds: list[NFTGameRound] = self.sql_handler.get_with_filter_and_order(
            query_filters=[col(NFTGameRound.start_time) > utcnow()],
            limit=1,
            order_by_column_name=NFTGameRound.start_time.key,  # type: ignore
            order_desc=False,
        )
        return next_rounds[0] if next_rounds else None

    def get_game_round_is_active(self) -> bool:
        round_ = self.get_current_round()
        return (
            round_ is not None
            # Needs to be set as restarted so we know the treasury, balances, etc. is set.
            and round_.started
        )

    def set_as_started(self, round_: NFTGameRound) -> None:
        round_.started = True
        self.sql_handler.save_multiple([round_])
