# inspired by crewAI's LongTermMemory (https://github.com/joaomdmoura/crewAI/blob/main/src/crewai/memory/long_term/long_term_memory.py)
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Sequence

from prediction_market_agent_tooling.deploy.agent import Answer
from prediction_market_agent_tooling.tools.utils import check_not_none, utcnow
from pydantic import BaseModel

from prediction_market_agent.db.db_storage import DBStorage
from prediction_market_agent.db.models import LongTermMemories


class ChatMessage(BaseModel):
    content: str
    role: str


class DatedChatMessage(ChatMessage):
    datetime_: datetime

    @staticmethod
    def from_long_term_memory(
        long_term_memory: LongTermMemories,
    ) -> "DatedChatMessage":
        metadata = json.loads(check_not_none(long_term_memory.metadata_))
        return DatedChatMessage(
            content=metadata["content"],
            role=metadata["role"],
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


class SimpleMemoryThinkThoroughly(BaseModel):
    metadata: AnswerWithScenario
    datetime_: datetime

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


class LongTermMemory:
    def __init__(self, task_description: str, sqlalchemy_db_url: str | None = None):
        self.task_description = task_description
        self.storage = DBStorage(sqlalchemy_db_url=sqlalchemy_db_url)

    def save_history(self, history: list[Dict[str, Any]]) -> None:
        """Save item to storage. Note that score allows many types for easier handling by agent."""

        history_items = [
            LongTermMemories(
                task_description=self.task_description,
                metadata_=json.dumps(history_item),
                datetime_=utcnow(),
            )
            for history_item in history
        ]

        self.storage.save_multiple(history_items)

    def save_answer_with_scenario(
        self, answer_with_scenario: AnswerWithScenario
    ) -> None:
        return self.save_history([answer_with_scenario.dict()])

    def search(
        self,
        from_: datetime | None = None,
        to: datetime | None = None,
    ) -> Sequence[LongTermMemories]:
        return self.storage.load_long_term_memories(
            task_description=self.task_description,
            from_=from_,
            to=to,
        )


class ChatHistory(BaseModel):
    """
    A collection of chat messages. Assumes that the chat messages are ordered by
    datetime.
    """

    chat_messages: Sequence[ChatMessage]

    def add_message(self, chat_message: ChatMessage) -> None:
        list(self.chat_messages).append(chat_message)

    @staticmethod
    def from_list_of_dicts(list_of_dicts: list[Dict[str, str]]) -> "ChatHistory":
        chat_messages = [
            ChatMessage(role=str(h["role"]), content=str(h["value"]))
            for h in list_of_dicts
        ]  # Enforce string typing
        return ChatHistory(chat_messages=chat_messages)

    @property
    def num_messages(self) -> int:
        return len(self.chat_messages)


class DatedChatHistory(ChatHistory):
    chat_messages: Sequence[DatedChatMessage]

    @property
    def start_time(self) -> datetime:
        return self.chat_messages[0].datetime_

    @property
    def end_time(self) -> datetime:
        return self.chat_messages[-1].datetime_

    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time

    @classmethod
    def from_long_term_memory(
        cls,
        long_term_memory: LongTermMemory,
        from_: datetime | None = None,
        to: datetime | None = None,
    ) -> "DatedChatHistory":
        memories = long_term_memory.search(from_=from_, to=to)

        # Sort memories by datetime
        memories = sorted(memories, key=lambda m: m.datetime_)
        chat_messages = [DatedChatMessage.from_long_term_memory(m) for m in memories]
        return cls(chat_messages=chat_messages)

    def cluster_by_datetime(
        self, max_minutes_between_messages: int
    ) -> list["DatedChatHistory"]:
        """
        Cluster chat messages by datetime. Each cluster will have messages that
        are within `minutes` of each other.
        """
        clusters: list[DatedChatHistory] = []
        chat_messages = list(self.chat_messages).copy()
        while chat_messages:
            current_datetime = chat_messages[0].datetime_
            current_cluster: list[DatedChatMessage] = []
            while chat_messages and chat_messages[
                0
            ].datetime_ < current_datetime + timedelta(
                minutes=max_minutes_between_messages
            ):
                current_cluster.append(chat_messages.pop(0))
            clusters.append(DatedChatHistory(chat_messages=current_cluster))
        return clusters
