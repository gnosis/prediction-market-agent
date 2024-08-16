from typing import Generator

import pytest

from prediction_market_agent.agents.goal_manager import EvaluatedGoal
from prediction_market_agent.db.evaluated_goal_table_handler import (
    EvaluatedGoalTableHandler,
)

SQLITE_DB_URL = "sqlite://"
TEST_AGENT_ID = "test_agent_id"


@pytest.fixture(scope="function")
def table_handler() -> Generator[EvaluatedGoalTableHandler, None, None]:
    """Creates a in-memory SQLite DB for testing"""
    table_handler = EvaluatedGoalTableHandler(
        sqlalchemy_db_url=SQLITE_DB_URL,
        agent_id=TEST_AGENT_ID,
    )
    yield table_handler


def test_save_load_evaluated_goal(table_handler: EvaluatedGoalTableHandler) -> None:
    evaluated_goal = EvaluatedGoal(
        prompt="abc",
        motivation="def",
        completion_criteria="ghi",
        is_complete=True,
        reasoning="jkl",
        output="mno",
    )
    model = evaluated_goal.to_model(agent_id=TEST_AGENT_ID)
    table_handler.save_evaluated_goal(model=model)

    loaded_model = table_handler.get_latest_evaluated_goal()
    assert loaded_model
    loaded_evaluated_goal = EvaluatedGoal.from_model(model=loaded_model)
    assert loaded_evaluated_goal == evaluated_goal


# TODO
# def test_load_latest_prompt(memory_prompt_handler: PromptTableHandler) -> None:
#     prompt_text_first = "prompt_text_first"
#     prompt_text_second = "prompt_text_second"

#     memory_prompt_handler.save_prompt(prompt_text_first)
#     memory_prompt_handler.save_prompt(prompt_text_second)

#     # assert latest prompt is there
#     result = memory_prompt_handler.fetch_latest_prompt()
#     assert result
#     # ignore timezone
#     assert result.prompt == prompt_text_second
