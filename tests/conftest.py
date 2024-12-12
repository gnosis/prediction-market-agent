from typing import Generator

import pytest
from prediction_market_agent_tooling.tools.db.db_manager import DBManager

from prediction_market_agent.db.blockchain_message_table_handler import (
    BlockchainMessageTableHandler,
)
from prediction_market_agent.db.evaluated_goal_table_handler import (
    EvaluatedGoalTableHandler,
)
from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemoryTableHandler,
)
from prediction_market_agent.db.prompt_table_handler import PromptTableHandler


@pytest.fixture(scope="session")
def long_term_memory_task_description() -> str:
    return "LONG_TERM_MEMORY_TEST"


@pytest.fixture(scope="session")
def prompt_test_session_identifier() -> str:
    return "TEST_SESSION_IDENTIFIER"


@pytest.fixture(scope="function")
def long_term_memory_table_handler(
    long_term_memory_task_description: str,
) -> Generator[LongTermMemoryTableHandler, None, None]:
    """Creates a in-memory SQLite DB for testing"""
    long_term_memory = LongTermMemoryTableHandler(
        task_description=long_term_memory_task_description,
        sqlalchemy_db_url="sqlite://",
    )
    yield long_term_memory
    reset_init_params_db_manager(long_term_memory.sql_handler.db_manager)


@pytest.fixture(scope="function")
def prompt_table_handler(
    prompt_test_session_identifier: str,
) -> Generator[PromptTableHandler, None, None]:
    """Creates a in-memory SQLite DB for testing"""
    prompt_handler = PromptTableHandler(
        sqlalchemy_db_url="sqlite://",
        session_identifier=prompt_test_session_identifier,
    )
    yield prompt_handler
    reset_init_params_db_manager(prompt_handler.sql_handler.db_manager)


@pytest.fixture(scope="function")
def memory_blockchain_handler() -> Generator[BlockchainMessageTableHandler, None, None]:
    """Creates a in-memory SQLite DB for testing"""
    prompt_handler = BlockchainMessageTableHandler(
        sqlalchemy_db_url="sqlite://",
    )
    yield prompt_handler
    reset_init_params_db_manager(prompt_handler.sql_handler.db_manager)


def reset_init_params_db_manager(db_manager: DBManager) -> None:
    db_manager._engine.dispose()
    db_manager._initialized = False
    db_manager.cache_table_initialized = {}


@pytest.fixture(scope="module")
def mocked_agent_id() -> Generator[str, None, None]:
    yield "test_agent_id"


@pytest.fixture(scope="function")
def evaluated_goal_table_handler(
    mocked_agent_id: str,
) -> Generator[EvaluatedGoalTableHandler, None, None]:
    """Creates a in-memory SQLite DB for testing"""
    table_handler = EvaluatedGoalTableHandler(
        sqlalchemy_db_url="sqlite://",
        agent_id=mocked_agent_id,
    )
    yield table_handler
    reset_init_params_db_manager(table_handler.sql_handler.db_manager)
