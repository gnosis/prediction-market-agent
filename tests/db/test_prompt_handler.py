from typing import Generator

import pytest

from prediction_market_agent.agents.microchain_agent.prompt_handler import PromptHandler

SQLITE_DB_URL = "sqlite://"
TEST_SESSION_IDENTIFIER = "test_session_identifier"


@pytest.fixture(scope="function")
def memory_prompt_handler() -> Generator[PromptHandler, None, None]:
    """Creates a in-memory SQLite DB for testing"""
    prompt_handler = PromptHandler(
        sqlalchemy_db_url=SQLITE_DB_URL, session_identifier=TEST_SESSION_IDENTIFIER
    )
    yield prompt_handler


def test_save_prompt(memory_prompt_handler: PromptHandler) -> None:
    prompt_text = "abc"

    memory_prompt_handler.save_prompt(prompt_text)
    # assert prompt is there
    result = memory_prompt_handler.fetch_latest_prompt(
        session_identifier=TEST_SESSION_IDENTIFIER
    )
    assert result
    assert result.prompt == prompt_text


def test_load_latest_prompt(memory_prompt_handler: PromptHandler) -> None:
    prompt_text_first = "prompt_text_first"
    prompt_text_second = "prompt_text_second"

    memory_prompt_handler.save_prompt(prompt_text_first)
    memory_prompt_handler.save_prompt(prompt_text_second)

    # assert latest prompt is there
    result = memory_prompt_handler.fetch_latest_prompt(TEST_SESSION_IDENTIFIER)
    assert result
    # ignore timezone
    assert result.prompt == prompt_text_second
