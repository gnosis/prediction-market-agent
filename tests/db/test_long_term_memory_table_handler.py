import json
from typing import Generator

import pytest
from prediction_market_agent_tooling.tools.utils import utcnow

from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemoryTableHandler,
)

SQLITE_DB_URL = "sqlite://"
TASK_DESCRIPTION = "test_task_description"


@pytest.fixture(scope="function")
def memory_long_term_memory_handler() -> (
    Generator[LongTermMemoryTableHandler, None, None]
):
    """Creates a in-memory SQLite DB for testing"""
    long_term_memory_table_handler = LongTermMemoryTableHandler(
        task_description=TASK_DESCRIPTION, sqlalchemy_db_url=SQLITE_DB_URL
    )
    yield long_term_memory_table_handler


def test_save_load_long_term_memory_item(
    memory_long_term_memory_handler: LongTermMemoryTableHandler,
) -> None:
    first_item = {"a1": "b"}
    memory_long_term_memory_handler.save_history([first_item])
    results = memory_long_term_memory_handler.search()
    assert len(results) == 1

    # Now test filtering based on datetime
    timestamp = utcnow()
    second_item = {"a2": "c"}
    memory_long_term_memory_handler.save_history([second_item])

    results = memory_long_term_memory_handler.search(to_=timestamp)
    assert len(results) == 1
    assert json.loads(str(results[0].metadata_)) == first_item

    results = memory_long_term_memory_handler.search(from_=timestamp)
    assert len(results) == 1
    assert json.loads(str(results[0].metadata_)) == second_item

    # Retrieve all
    assert len(memory_long_term_memory_handler.search()) == 2
