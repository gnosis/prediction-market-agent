# inspired by crewAI's LongTermMemory (https://github.com/joaomdmoura/crewAI/blob/main/src/crewai/memory/long_term/long_term_memory.py)
from datetime import datetime
from typing import Any, Dict, Sequence

from prediction_market_agent.db.db_storage import DBStorage
from prediction_market_agent.db.models import LongTermMemories


# In the future, create a base class which this class extends.
class LongTermMemory:
    def __init__(self, task_description: str):
        self.task_description = task_description
        self.storage = DBStorage()

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
    ) -> Sequence[LongTermMemories]:
        return self.storage.load(
            task_description=self.task_description,
            from_=from_,
            to=to,
        )
