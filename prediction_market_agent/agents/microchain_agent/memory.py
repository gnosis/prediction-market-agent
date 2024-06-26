# inspired by crewAI's LongTermMemory (https://github.com/joaomdmoura/crewAI/blob/main/src/crewai/memory/long_term/long_term_memory.py)
import json
from datetime import datetime, timedelta
from typing import Dict, Sequence

from prediction_market_agent_tooling.tools.utils import check_not_none
from pydantic import BaseModel

from prediction_market_agent.agents.microchain_agent.answer_with_scenario import (
    AnswerWithScenario,
)
from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemoryTableHandler,
)
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


class ChatHistory(BaseModel):
    """
    A collection of chat messages. Assumes that the chat messages are ordered by
    datetime.
    """

    chat_messages: Sequence[ChatMessage]

    def add_message(self, chat_message: ChatMessage) -> None:
        list(self.chat_messages).append(chat_message)

    def save_to(self, long_term_memory: LongTermMemoryTableHandler) -> None:
        long_term_memory.save_history([m.model_dump() for m in self.chat_messages])

    @staticmethod
    def from_list_of_dicts(list_of_dicts: list[Dict[str, str]]) -> "ChatHistory":
        chat_messages = [
            ChatMessage(role=h["role"], content=h["content"]) for h in list_of_dicts
        ]
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
        long_term_memory: LongTermMemoryTableHandler,
        from_: datetime | None = None,
        to: datetime | None = None,
    ) -> "DatedChatHistory":
        memories = long_term_memory.search(from_=from_, to_=to)

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

    def to_undated_chat_history(self) -> ChatHistory:
        # Convert DatedChatMessages to ChatMessages
        chat_messages = [
            ChatMessage(content=m.content, role=m.role) for m in self.chat_messages
        ]
        return ChatHistory(chat_messages=chat_messages)

    def save_to(self, long_term_memory: LongTermMemoryTableHandler) -> None:
        self.to_undated_chat_history().save_to(long_term_memory)
