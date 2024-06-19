import json
from typing import Generator

import pytest
from prediction_market_agent_tooling.tools.utils import utcnow
from sqlmodel import Session

from prediction_market_agent.db.sql_handler import SqlHandler
from prediction_market_agent.db.models import LongTermMemories, Prompt


@pytest.fixture(scope="function")
def prompt_sql_handler() -> Generator[SqlHandler[Prompt], None, None]:
    sql_handler = SqlHandler[Prompt](model=Prompt, sqlalchemy_db_url="sqlite://")
    sql_handler._init_table_if_not_exists()
    yield sql_handler


def test_load_all(prompt_sql_handler: SqlHandler[Prompt]) -> None:
    assert len(prompt_sql_handler.select_all()) == 0
    p = Prompt(prompt="hello", datetime_=utcnow(), session_identifier="b")
    with Session(prompt_sql_handler.engine) as session:
        session.add(p)
        session.commit()
    assert len(prompt_sql_handler.select_all()) == 1
