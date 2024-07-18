from datetime import datetime
from typing import Generator

import pytest
from prediction_market_agent_tooling.tools.utils import utcnow

from prediction_market_agent.agents.microchain_agent.memory import (
    DatedChatHistory,
    DatedChatMessage,
)
from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemoryTableHandler,
)


@pytest.fixture(scope="session")
def long_term_memory() -> Generator[LongTermMemoryTableHandler, None, None]:
    """Creates a in-memory SQLite DB for testing"""
    long_term_memory = LongTermMemoryTableHandler(
        task_description="test", sqlalchemy_db_url="sqlite://"
    )
    yield long_term_memory


@pytest.fixture(scope="session")
def chat_history() -> DatedChatHistory:
    chat_messages = [
        DatedChatMessage(
            content="foo", role="system", datetime_=datetime(2022, 1, 1, 0, 0)
        ),
        DatedChatMessage(
            content="bar", role="assistant", datetime_=datetime(2022, 1, 1, 0, 20)
        ),
        DatedChatMessage(
            content="baz", role="user", datetime_=datetime(2022, 1, 1, 0, 25)
        ),
        DatedChatMessage(
            content="qux", role="system", datetime_=datetime(2022, 1, 1, 0, 30)
        ),
        DatedChatMessage(
            content="quux", role="assistant", datetime_=datetime(2022, 1, 1, 0, 35)
        ),
    ]
    return DatedChatHistory(chat_messages=chat_messages)


def test_chat_history_clustering(chat_history: DatedChatHistory) -> None:
    assert chat_history.num_messages == 5

    clusters = chat_history.cluster_by_session()
    assert len(clusters) == 2
    assert clusters[0].num_messages == 3
    assert clusters[1].num_messages == 2

    # Check that each cluster starts with a system message
    for cluster in clusters:
        assert cluster.chat_messages[0].is_system_message

    # Check that chat_history is still the same length after clustering
    assert chat_history.num_messages == 5


def test_save_to_and_load_from_memory(
    long_term_memory: LongTermMemoryTableHandler, chat_history: DatedChatHistory
) -> None:
    datetime_now = utcnow()
    chat_history.save_to(long_term_memory)
    new_chat_history = DatedChatHistory.from_long_term_memory(
        long_term_memory=long_term_memory,
        from_=datetime_now,
    )
    assert (
        new_chat_history.to_undated_chat_history()
        == chat_history.to_undated_chat_history()
    )
