import datetime
import json
from typing import Generator

import pytest
from prediction_market_agent_tooling.tools.utils import utcnow

from prediction_market_agent.db.db_storage import DBStorage
from prediction_market_agent.db.models import (
    PROMPT_DEFAULT_SESSION_IDENTIFIER,
    LongTermMemories,
    Prompt,
)

SQLITE_DB_URL = "sqlite://"


@pytest.fixture(scope="function")
def db_storage_test() -> Generator[DBStorage, None, None]:
    """Creates a in-memory SQLite DB for testing"""
    db_storage = DBStorage(sqlalchemy_db_url=SQLITE_DB_URL)
    db_storage._initialize_db()
    yield db_storage


def test_db_storage_connects(db_storage_test: DBStorage) -> None:
    # assert table is there
    conn = db_storage_test.engine.connect()
    assert conn


def test_save_load_long_term_memory_item(db_storage_test: DBStorage) -> None:
    task_description = "test"
    first_item = {"a1": "b"}
    long_term_memory = LongTermMemories(
        task_description=task_description,
        metadata_=json.dumps(first_item),
        datetime_=utcnow(),
    )

    db_storage_test.save_multiple([long_term_memory])
    results = db_storage_test.load_long_term_memories(task_description)
    assert len(results) == 1

    # Now test filtering based on datetime
    timestamp = utcnow()
    second_item = {"a2": "c"}
    db_storage_test.save_multiple_long_term_memories(task_description, [second_item])

    results = db_storage_test.load_long_term_memories(
        task_description=task_description, to=timestamp
    )
    assert len(results) == 1
    assert json.loads(str(results[0].metadata_)) == first_item

    results = db_storage_test.load_long_term_memories(
        task_description=task_description, from_=timestamp
    )
    assert len(results) == 1
    assert json.loads(str(results[0].metadata_)) == second_item

    assert (
        len(db_storage_test.load_long_term_memories(task_description=task_description))
        == 2
    )


def test_save_prompt(db_storage_test: DBStorage) -> None:
    prompt_text = "abc"
    prompt = Prompt(
        prompt=prompt_text,
        session_identifier=PROMPT_DEFAULT_SESSION_IDENTIFIER,
        datetime_=utcnow(),
    )
    db_storage_test.save_multiple([prompt])
    # assert prompt is there
    result = db_storage_test.load_latest_prompt(
        session_identifier=PROMPT_DEFAULT_SESSION_IDENTIFIER
    )
    assert result
    assert result.prompt == prompt_text


def test_load_latest_prompt(db_storage_test: DBStorage) -> None:
    prompt_text = "abc"
    # We ignore the timezone for testing purposes
    older_timestamp = datetime.datetime.now()
    newer_timestamp = datetime.datetime.now()
    session_identifier = "whatever"
    prompt = Prompt(
        prompt=prompt_text,
        session_identifier=session_identifier,
        datetime_=older_timestamp,
    )
    db_storage_test.save_multiple([prompt])
    prompt = Prompt(
        prompt=prompt_text,
        session_identifier=session_identifier,
        datetime_=newer_timestamp,
    )
    db_storage_test.save_multiple([prompt])
    # assert latest prompt is there
    result = db_storage_test.load_latest_prompt(session_identifier=session_identifier)
    assert result
    # ignore timezone
    assert result.datetime_ == newer_timestamp
