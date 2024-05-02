from decimal import Decimal
from decimal import Decimal
from typing import Sequence

from loguru import logger
from sqlmodel import create_engine, SQLModel, Session, select

from prediction_market_agent.db.models import LongTermMemories
from prediction_market_agent.utils import DBKeys


class DBStorage:
    def __init__(self):
        keys = DBKeys()
        if not keys.sqlalchemy_db_url:
            raise EnvironmentError(
                "Cannot initialize DBHandler without a valid sqlalchemy_db_url"
            )
        self.engine = create_engine(keys.sqlalchemy_db_url)

    def _initialize_db(self):
        """
        Creates the tables if they don't exist
        """
        try:
            # trick for making models import mandatory - models must be imported for metadata.create_all to work
            logger.debug(f"tables being added {LongTermMemories}")
            SQLModel.metadata.create_all(self.engine)
        except Exception as e:
            logger.warning("Could not create table(s) ", e)

    def save(
        self,
        task_description: str,
        metadata_: str,
        score: Decimal,
    ) -> None:
        """Saves data to the LTM table with error handling."""

        with Session(self.engine) as session:
            long_term_memory_item = LongTermMemories(
                task_description=task_description, metadata_=metadata_, score=score
            )
            session.add(long_term_memory_item)
            session.commit()

    def load(
        self, task_description: str, latest_n: int = 5
    ) -> Sequence[LongTermMemories]:
        """Queries the LTM table by task description with error handling."""
        key = "task_description"
        with Session(self.engine) as session:
            items = session.exec(
                select(LongTermMemories)
                .where(LongTermMemories.task_description == task_description)
                .limit(latest_n)
            ).all()
            return items
