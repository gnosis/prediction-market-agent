from typing import Generator

import pytest
from prediction_market_agent_tooling.tools.db.db_manager import DBManager

from prediction_market_agent.db.evaluated_goal_table_handler import (
    EvaluatedGoalTableHandler,
)
from prediction_market_agent.db.models import Prompt
from prediction_market_agent.db.sql_handler import SQLHandler


@pytest.fixture(scope="function")
def prompt_sql_handler_in_memory() -> Generator[SQLHandler, None, None]:
    sql_handler = SQLHandler(model=Prompt, sqlalchemy_db_url="sqlite:///:memory:")
    sql_handler._init_table_if_not_exists()
    yield sql_handler
    # We need to reset the initialization parameters for isolation between tests
    reset_init_params_db_manager(sql_handler.db_manager)


def reset_init_params_db_manager(db_manager: DBManager) -> None:
    db_manager._engine.dispose()
    db_manager._initialized = False
    db_manager.cache_table_initialized = {}


@pytest.fixture(scope="module")
def mocked_agent_id() -> Generator[str, None, None]:
    yield "test_agent_id"


@pytest.fixture(scope="function")
def table_handler(
    mocked_agent_id: str,
) -> Generator[EvaluatedGoalTableHandler, None, None]:
    """Creates a in-memory SQLite DB for testing"""
    table_handler = EvaluatedGoalTableHandler(
        sqlalchemy_db_url="sqlite:///:memory:",
        agent_id=mocked_agent_id,
    )
    yield table_handler
    reset_init_params_db_manager(table_handler.sql_handler.db_manager)
