# inspired by crewAI's LongTermMemory (https://github.com/joaomdmoura/crewAI/blob/main/src/crewai/memory/long_term/long_term_memory.py)
from typing import Dict, Any

from prediction_market_agent.db.db_storage import DBStorage


class Memory:
    def __init__(self, storage: DBStorage):
        self.storage = storage

    def save(self, score: float, metadata: dict[str, str]) -> None:
        raise NotImplementedError("To be implemented in subclass")

    def search(self, task: str, latest_n: int = 3) -> list[Dict[str, Any]]:
        raise NotImplementedError("To be implemented in subclass")


# In the future, create a base class which this class extends.
class LongTermMemory(Memory):
    def __init__(self, task_description: str, storage: DBStorage):
        self.task_description = task_description
        super().__init__(storage)

    def save(self, score: float, metadata: dict[str, str]) -> None:
        self.storage.save(
            task_description=self.task_description,
            score=score,
            metadata=metadata,
        )

    def search(self, latest_n: int = 3) -> list[Dict[str, Any]]:
        return [i.dict() for i in self.storage.load(self.task_description, latest_n)]
