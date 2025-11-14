import time

from prediction_market_agent.db.prompt_table_handler import PromptTableHandler

SQLITE_DB_URL = "sqlite://"
TEST_SESSION_IDENTIFIER = "test_session_identifier"


def test_save_prompt(prompt_table_handler: PromptTableHandler) -> None:
    prompt_text = "abc"

    prompt_table_handler.save_prompt(prompt_text)
    # assert prompt is there
    result = prompt_table_handler.fetch_latest_prompt()
    assert result
    assert result.prompt == prompt_text


def test_load_latest_prompt(prompt_table_handler: PromptTableHandler) -> None:
    prompt_text_first = "prompt_text_first"
    prompt_text_second = "prompt_text_second"

    prompt_table_handler.save_prompt(prompt_text_first)
    time.sleep(0.1)
    prompt_table_handler.save_prompt(prompt_text_second)

    # assert latest prompt is there
    result = prompt_table_handler.fetch_latest_prompt()
    assert result
    # ignore timezone
    assert result.prompt == prompt_text_second
