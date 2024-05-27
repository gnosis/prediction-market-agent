# inspired by crewAI's LongTermMemory (https://github.com/joaomdmoura/crewAI/blob/main/src/crewai/memory/long_term/long_term_memory.py)
import json
from datetime import datetime
from typing import Any, Dict

from prediction_market_agent_tooling.tools.utils import check_not_none
from pydantic import BaseModel

from prediction_market_agent.db.db_storage import DBStorage
from prediction_market_agent.db.models import LongTermMemories


class SimpleMemory(BaseModel):
    content: str
    datetime_: datetime

    @staticmethod
    def from_long_term_memory(long_term_memory: LongTermMemories) -> "SimpleMemory":
        return SimpleMemory(
            content=json.loads(check_not_none(long_term_memory.metadata_))["content"],
            datetime_=long_term_memory.datetime_,
        )

    def __str__(self) -> str:
        return f"{self.datetime_}: {self.content}"


class LongTermMemory:
    def __init__(self, task_description: str, sqlalchemy_db_url: str | None = None):
        self.task_description = task_description
        self.storage = DBStorage(sqlalchemy_db_url=sqlalchemy_db_url)

    def save_history(self, history: list[Dict[str, Any]]) -> None:
        """Save item to storage. Note that score allows many types for easier handling by agent."""
        self.storage.save_multiple(
            task_description=self.task_description,
            history=history,
        )

    def search(
        self,
        from_: datetime | None = None,
        to: datetime | None = None,
    ) -> list[SimpleMemory]:
        long_term_memories = self.storage.load(
            task_description=self.task_description,
            from_=from_,
            to=to,
        )
        return [SimpleMemory.from_long_term_memory(ltm) for ltm in long_term_memories]
