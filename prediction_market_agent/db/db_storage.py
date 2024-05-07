import json
from typing import Any, Dict, Sequence

from loguru import logger
from prediction_market_agent_tooling.tools.utils import utcnow, check_not_none
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
        self, task_description: str, latest_n: int = 5
    ) -> Sequence[LongTermMemories]:
        """Queries the LTM table by task description with error handling."""
        with Session(self.engine) as session:
            items = session.exec(
                select(LongTermMemories)
                .where(LongTermMemories.task_description == task_description)
                .order_by(desc(LongTermMemories.datetime_))
                .limit(latest_n)
            ).all()
            return items
