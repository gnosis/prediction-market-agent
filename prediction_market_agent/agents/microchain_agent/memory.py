# inspired by crewAI's LongTermMemory (https://github.com/joaomdmoura/crewAI/blob/main/src/crewai/memory/long_term/long_term_memory.py)
import json
from abc import ABC, abstractmethod
from datetime import datetime

from prediction_market_agent_tooling.deploy.agent import Answer
from prediction_market_agent_tooling.tools.utils import check_not_none, utcnow
from pydantic import BaseModel

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
            metadata=AnswerWithScenario.model_validate_json(
                check_not_none(long_term_memory.metadata_)
            ),
            datetime_=long_term_memory.datetime_,
        )
