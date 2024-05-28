# inspired by crewAI's LongTermMemory (https://github.com/joaomdmoura/crewAI/blob/main/src/crewai/memory/long_term/long_term_memory.py)
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Sequence

from prediction_market_agent_tooling.deploy.agent import Answer
from prediction_market_agent_tooling.tools.utils import check_not_none
from pydantic import BaseModel

from prediction_market_agent.db.db_storage import DBStorage
from prediction_market_agent.db.models import LongTermMemories


class MemoryContainer(BaseModel, ABC):
    datetime_: datetime

    @staticmethod
    @abstractmethod
    def from_long_term_memory(
        long_term_memory: LongTermMemories,
    ) -> "MemoryContainer":
        pass


class SimpleMemoryMicrochain(MemoryContainer):
    content: str

    @staticmethod
    def from_long_term_memory(
        long_term_memory: LongTermMemories,
    ) -> "SimpleMemoryMicrochain":
        return SimpleMemoryMicrochain(
            content=json.loads(check_not_none(long_term_memory.metadata_))["content"],
            datetime_=long_term_memory.datetime_,
        )

    def __str__(self) -> str:
        return f"{self.datetime_}: {self.content}"


class AnswerWithScenario(Answer):
    scenario: str
    question: str

    @staticmethod
    def build_from_answer(
        answer: Answer, scenario: str, question: str
    ) -> "AnswerWithScenario":
        return AnswerWithScenario(scenario=scenario, question=question, **answer.dict())


class SimpleMemoryThinkThoroughly(MemoryContainer):
    metadata: AnswerWithScenario

    @staticmethod
    def from_long_term_memory(
        long_term_memory: LongTermMemories,
    ) -> "SimpleMemoryThinkThoroughly":
        return SimpleMemoryThinkThoroughly(
            metadata=AnswerWithScenario.model_validate_json(long_term_memory.metadata_),
            datetime_=long_term_memory.datetime_,
        )


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

    def save_answer_with_scenario(
        self, answer_with_scenario: AnswerWithScenario
    ) -> None:
        self.storage.save_multiple(
            task_description=self.task_description,
            history=[answer_with_scenario.dict()],
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
