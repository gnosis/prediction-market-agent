import json
from typing import Generator

import pytest
from prediction_market_agent_tooling.tools.utils import utcnow

from prediction_market_agent.db.db_storage import DBStorage


@pytest.fixture(scope="session")
def db_storage_test() -> Generator[DBStorage, None, None]:
    """Creates a in-memory SQLite DB for testing"""
    db_storage = DBStorage(sqlalchemy_db_url="sqlite://")
    db_storage._initialize_db()
    yield db_storage


def test_db_storage_connects(db_storage_test: DBStorage) -> None:
    # assert table is there
    conn = db_storage_test.engine.connect()
    assert conn


def test_save_load_long_term_memory_item(db_storage_test: DBStorage) -> None:
    task_description = "test"
    first_item = {"a1": "b"}
    db_storage_test.save_multiple(task_description, [first_item])
    results = db_storage_test.load(task_description)
    assert len(results) == 1

    # Now test filtering based on datetime
    timestamp = utcnow()
    second_item = {"a2": "c"}
    db_storage_test.save_multiple(task_description, [second_item])

    results = db_storage_test.load(task_description=task_description, to=timestamp)
    assert len(results) == 1
    assert json.loads(results[0].metadata_) == first_item

    results = db_storage_test.load(task_description=task_description, from_=timestamp)
    assert len(results) == 1
    assert json.loads(results[0].metadata_) == second_item

    assert len(db_storage_test.load(task_description=task_description)) == 2
