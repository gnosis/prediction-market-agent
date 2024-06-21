from datetime import datetime
from typing import Generator

import pytest

from prediction_market_agent.agents.microchain_agent.memory import (
    DatedChatHistory,
    DatedChatMessage,
    LongTermMemory,
)


@pytest.fixture(scope="session")
def long_term_memory() -> Generator[LongTermMemory, None, None]:
    """Creates a in-memory SQLite DB for testing"""
    long_term_memory = LongTermMemory(
        task_description="test", sqlalchemy_db_url="sqlite://"
    )
    long_term_memory.storage._initialize_db()
    yield long_term_memory


@pytest.fixture(scope="session")
def chat_history() -> DatedChatHistory:
    chat_messages = [
        DatedChatMessage(
            content="foo", role="user", datetime_=datetime(2022, 1, 1, 0, 0)
        ),
        DatedChatMessage(
            content="bar", role="agent", datetime_=datetime(2022, 1, 1, 0, 20)
        ),
        DatedChatMessage(
            content="baz", role="user", datetime_=datetime(2022, 1, 1, 0, 25)
        ),
    ]
    return DatedChatHistory(chat_messages=chat_messages)


def test_chat_history_clustering(chat_history: DatedChatHistory) -> None:
    assert chat_history.num_messages == 3

    clusters0 = chat_history.cluster_by_datetime(max_minutes_between_messages=10)
    assert len(clusters0) == 2
    assert clusters0[0].num_messages == 1
    assert clusters0[1].num_messages == 2

    clusters1 = chat_history.cluster_by_datetime(max_minutes_between_messages=30)
    assert len(clusters1) == 1
    assert clusters1[0].num_messages == 3

    # Check that chat_history is still the same length after clustering
    assert chat_history.num_messages == 3


def test_save_to_memory(
    long_term_memory: LongTermMemory, chat_history: DatedChatHistory
) -> None:
    datetime_now = datetime.now()
    chat_history.save_to(long_term_memory)
    breakpoint()
    assert long_term_memory.search(from_=datetime_now)
