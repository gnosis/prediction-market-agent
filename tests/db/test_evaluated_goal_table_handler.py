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


def test_save_load_evaluated_goal_0(table_handler: EvaluatedGoalTableHandler) -> None:
    evaluated_goal = EvaluatedGoal(
        goal="abc",
        motivation="def",
        completion_criteria="ghi",
        is_complete=True,
        reasoning="jkl",
        output="mno",
    )
    table_handler.save_evaluated_goal(
        model=evaluated_goal.to_model(agent_id=TEST_AGENT_ID)
    )

    loaded_models = table_handler.get_latest_evaluated_goals(limit=1)
    assert len(loaded_models) == 1
    loaded_evaluated_goal = EvaluatedGoal.from_model(model=loaded_models[0])
    assert loaded_evaluated_goal == evaluated_goal


def test_save_load_evaluated_goal_1(table_handler: EvaluatedGoalTableHandler) -> None:
    evaluated_goal0 = EvaluatedGoal(
        goal="foo",
        motivation="foo",
        completion_criteria="foo",
        is_complete=True,
        reasoning="foo",
        output="foo",
    )
    evaluated_goal1 = EvaluatedGoal(
        goal="bar",
        motivation="bar",
        completion_criteria="bar",
        is_complete=False,
        reasoning="bar",
        output="bar",
    )

    table_handler.save_evaluated_goal(
        model=evaluated_goal0.to_model(agent_id=TEST_AGENT_ID)
    )
    table_handler.save_evaluated_goal(
        model=evaluated_goal1.to_model(agent_id=TEST_AGENT_ID)
    )

    loaded_models = table_handler.get_latest_evaluated_goals(limit=1)
    assert len(loaded_models) == 1
    loaded_evaluated_goal = EvaluatedGoal.from_model(model=loaded_models[0])
    assert loaded_evaluated_goal == evaluated_goal1

    for limit in [2, 3]:
        loaded_models = table_handler.get_latest_evaluated_goals(limit=limit)
        assert len(loaded_models) == 2
        # Check LIFO order
        assert loaded_models[0].datetime_ > loaded_models[1].datetime_
        assert [EvaluatedGoal.from_model(model) for model in loaded_models] == [
            evaluated_goal1,
            evaluated_goal0,
        ]


def test_save_load_evaluated_goal_multiple_agents(
    table_handler: EvaluatedGoalTableHandler,
) -> None:
    evaluated_goal0 = EvaluatedGoal(
        goal="foo",
        motivation="foo",
        completion_criteria="foo",
        is_complete=True,
        reasoning="foo",
        output="foo",
    )
    evaluated_goal1 = EvaluatedGoal(
        goal="bar",
        motivation="bar",
        completion_criteria="bar",
        is_complete=False,
        reasoning="bar",
        output="bar",
    )

    table_handler.save_evaluated_goal(
        model=evaluated_goal0.to_model(agent_id=TEST_AGENT_ID)
    )
    table_handler.save_evaluated_goal(
        model=evaluated_goal1.to_model(agent_id=TEST_AGENT_ID + "1")
    )

    loaded_models = table_handler.get_latest_evaluated_goals(limit=1)
    assert len(loaded_models) == 1
    loaded_evaluated_goal = EvaluatedGoal.from_model(model=loaded_models[0])
    assert loaded_evaluated_goal == evaluated_goal0
