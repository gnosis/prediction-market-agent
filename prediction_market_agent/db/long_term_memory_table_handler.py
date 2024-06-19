import json
import typing as t
from datetime import datetime

from prediction_market_agent_tooling.tools.utils import utcnow
from sqlmodel import col

from prediction_market_agent.agents.microchain_agent.memory import AnswerWithScenario
from prediction_market_agent.db.models import LongTermMemories
from prediction_market_agent.db.sql_handler import SQLHandler


class LongTermMemoryTableHandler:
    def __init__(self, task_description: str, sqlalchemy_db_url: str | None = None):
        self.task_description = task_description
        self.sql_handler = SQLHandler(
            model=LongTermMemories, sqlalchemy_db_url=sqlalchemy_db_url
        )

    def save_history(self, history: list[dict[str, t.Any]]) -> None:
        """Save item to storage. Note that score allows many types for easier handling by agent."""

        history_items = [
            LongTermMemories(
                task_description=self.task_description,
                metadata_=json.dumps(history_item),
                datetime_=utcnow(),
            )
            for history_item in history
        ]

        self.sql_handler.save_multiple(history_items)

    def save_answer_with_scenario(
        self, answer_with_scenario: AnswerWithScenario
    ) -> None:
        return self.save_history([answer_with_scenario.dict()])

    def search(
        self,
        from_: datetime | None = None,
        to_: datetime | None = None,
    ) -> t.Sequence[LongTermMemories]:
        """Searches the LongTermMemoryTableHandler for entries within a specified datetime range that match
        self.task_description."""
        query_filters = [
            col(LongTermMemories.task_description) == self.task_description
        ]
        if from_ is not None:
            query_filters.append(col(LongTermMemories.datetime_) >= from_)
        if to_ is not None:
            query_filters.append(col(LongTermMemories.datetime_) <= to_)

        return self.sql_handler.get_with_filter_and_order(
            query_filters=query_filters,
            order_by_column_name=LongTermMemories.datetime_.key,  # type: ignore
            order_desc=True,
        )
