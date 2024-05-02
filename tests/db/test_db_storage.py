import pytest
from sqlmodel import SQLModel, create_engine
from prediction_market_agent.db.db_storage import DBStorage


@pytest.fixture(scope="session")
def db_storage_test():
    """Creates a in-memory SQLite DB for testing"""
    db_storage = DBStorage(sqlalchemy_db_url="sqlite://")
    db_storage._initialize_db()
    yield db_storage


def test_db_storage_connects(db_storage_test: DBStorage):
    # assert table is there
    conn = db_storage_test.engine.connect()
    assert conn


def test_save_load_long_term_memory_item(db_storage_test: DBStorage):
    task_description = "test"
    db_storage_test.save_multiple(task_description, [{"a1": "b"}])
    results = db_storage_test.load(task_description)
    assert len(results) == 1
