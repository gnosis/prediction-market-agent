import json
import typing as t

from prediction_market_agent_tooling.tools.utils import DatetimeUTC, utcnow
from sqlalchemy.sql.elements import ColumnElement
from sqlmodel import col

from prediction_market_agent.agents.identifiers import AgentIdentifier
from prediction_market_agent.agents.microchain_agent.answer_with_scenario import (
    AnswerWithScenario,
)
from prediction_market_agent.db.models import LongTermMemories
from prediction_market_agent.db.sql_handler import SQLHandler


class LongTermMemoryTableHandler:
    def __init__(self, task_description: str, sqlalchemy_db_url: str | None = None):
        self.task_description = task_description
        self.sql_handler = SQLHandler(
            model=LongTermMemories, sqlalchemy_db_url=sqlalchemy_db_url
        )

    @staticmethod
    def from_agent_identifier(
        identifier: AgentIdentifier,
    ) -> "LongTermMemoryTableHandler":
        return LongTermMemoryTableHandler(task_description=identifier)

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
        return self.save_history([answer_with_scenario.model_dump()])

    def _get_query_filters(
        self, from_: DatetimeUTC | None, to_: DatetimeUTC | None
    ) -> list[ColumnElement[bool]]:
        query_filters = [
            col(LongTermMemories.task_description) == self.task_description
        ]
        if from_ is not None:
            query_filters.append(col(LongTermMemories.datetime_) >= from_)
        if to_ is not None:
            query_filters.append(col(LongTermMemories.datetime_) <= to_)
        return query_filters

    def search(
        self,
        from_: DatetimeUTC | None = None,
        to_: DatetimeUTC | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[LongTermMemories]:
        """Searches the LongTermMemoryTableHandler for entries within a specified datetime range that match
        self.task_description."""
        query_filters = self._get_query_filters(from_, to_)
        return self.sql_handler.get_with_filter_and_order(
            query_filters=query_filters,
            order_by_column_name=LongTermMemories.datetime_.key,  # type: ignore[attr-defined]
            order_desc=True,
            offset=offset,
            limit=limit,
        )

    def count(self) -> int:
        query_filters = self._get_query_filters(None, None)
        return self.sql_handler.count(query_filters=query_filters)
