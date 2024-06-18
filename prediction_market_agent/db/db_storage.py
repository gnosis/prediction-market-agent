import json
from datetime import datetime
from typing import Any, Dict, Sequence

from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.utils import check_not_none, utcnow
from sqlmodel import Session, SQLModel, create_engine, desc, select

from prediction_market_agent.db.models import (
    PROMPT_DEFAULT_SESSION_IDENTIFIER,
    LongTermMemories,
    Prompt,
)
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
        logger.debug(f"tables being added {LongTermMemories} {Prompt}")
        SQLModel.metadata.create_all(self.engine)

    def save_multiple(self, items: list[Any]) -> None:
        with Session(self.engine) as session:
            session.add_all(items)
            session.commit()

    def save_multiple_long_term_memories(
        self,
        task_description: str,
        history: list[Dict[str, Any]],
    ) -> None:
        """Saves data to the LTM table with error handling."""

        history_items = [
            LongTermMemories(
                task_description=task_description,
                metadata_=json.dumps(history_item),
                datetime_=utcnow(),
            )
            for history_item in history
        ]

        self.save_multiple(history_items)

    def load_latest_prompt(self, session_identifier: str) -> Prompt | None:
        """
        Queries the prompts table by session identifier with error handling
        """
        with Session(self.engine) as session:
            query = (
                select(Prompt)
                .where(Prompt.session_identifier == session_identifier)
                .order_by(desc(Prompt.datetime_))
            )
            result = session.exec(query).first()
            return result

    def load_long_term_memories(
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

            return session.exec(query.order_by(desc(LongTermMemories.datetime_))).all()
