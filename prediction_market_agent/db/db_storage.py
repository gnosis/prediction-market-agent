import json
from datetime import datetime
from typing import Any, Dict, Sequence

from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.utils import check_not_none, utcnow
from sqlmodel import Session, SQLModel, create_engine, desc, select

from prediction_market_agent.db.models import LongTermMemories
from prediction_market_agent.utils import DBKeys


class DBStorage:
    def __init__(self, sqlalchemy_db_url: str | None = None):
        self.engine = create_engine(
            sqlalchemy_db_url
            if sqlalchemy_db_url
            else check_not_none(DBKeys().SQLALCHEMY_DB_URL)
        )

    def _initialize_db(self) -> None:
        """
        Creates the tables if they don't exist
        """

        # trick for making models import mandatory - models must be imported for metadata.create_all to work
        logger.debug(f"tables being added {LongTermMemories}")
        SQLModel.metadata.create_all(self.engine)

    def save_multiple(
        self,
        task_description: str,
        history: list[Dict[str, Any]],
    ) -> None:
        """Saves data to the LTM table with error handling."""

        with Session(self.engine) as session:
            for history_item in history:
                long_term_memory_item = LongTermMemories(
                    task_description=task_description,
                    metadata_=json.dumps(history_item),
                    datetime_=utcnow(),
                )
                session.add(long_term_memory_item)
            session.commit()

    def load(
        self,
        task_description: str,
        from_: datetime | None = None,
        to: datetime | None = None,
    ) -> Sequence[LongTermMemories]:
        """
        Queries the LTM table by task description within a specific datetime
        range, with error handling.
        """
        with Session(self.engine) as session:
            query = select(LongTermMemories).where(
                LongTermMemories.task_description == task_description
            )
            if from_ is not None:
                query = query.where(LongTermMemories.datetime_ >= from_)
            if to is not None:
                query = query.where(LongTermMemories.datetime_ <= to)

            items = session.exec(query.order_by(desc(LongTermMemories.datetime_))).all()
            return items
