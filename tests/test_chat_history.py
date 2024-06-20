from datetime import datetime

from prediction_market_agent.agents.microchain_agent.memory import (
    DatedChatHistory,
    DatedChatMessage,
)


def test_chat_history_clustering() -> None:
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
    chat_history = DatedChatHistory(chat_messages=chat_messages)
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
