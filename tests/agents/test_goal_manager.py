import pytest

from prediction_market_agent.agents.goal_manager import EvaluatedGoal, GoalManager
from tests.utils import RUN_PAID_TESTS


def test_have_reached_retry_limit() -> None:
    goal_manager = GoalManager(
        agent_id="test_agent",
        high_level_description="foo",
        agent_capabilities="bar",
        retry_limit=0,
    )

    g0 = EvaluatedGoal(
        goal="goal0",
        motivation="motivation",
        completion_criteria="completion_criteria",
        is_complete=False,
        reasoning="reasoning",
        output=None,
    )
    g1 = g0.model_copy()
    g1.goal = "goal1"

    assert goal_manager.have_reached_retry_limit(latest_evaluated_goals=[]) is True

    goal_manager.retry_limit = 1
    assert goal_manager.have_reached_retry_limit(latest_evaluated_goals=[]) is False
    assert goal_manager.have_reached_retry_limit(latest_evaluated_goals=[g0]) is False
    assert (
        goal_manager.have_reached_retry_limit(latest_evaluated_goals=[g0, g0]) is True
    )

    goal_manager.retry_limit = 2
    assert goal_manager.have_reached_retry_limit(latest_evaluated_goals=[]) is False
    assert goal_manager.have_reached_retry_limit(latest_evaluated_goals=[g0]) is False
    assert (
        goal_manager.have_reached_retry_limit(latest_evaluated_goals=[g0, g0]) is False
    )
    assert (
        goal_manager.have_reached_retry_limit(latest_evaluated_goals=[g0, g0, g0])
        is True
    )
    assert (
        goal_manager.have_reached_retry_limit(latest_evaluated_goals=[g0, g0, g1])
        is False
    )
    assert (
        goal_manager.have_reached_retry_limit(latest_evaluated_goals=[g0, g0, g0, g1])
        is True
    )
    assert (
        goal_manager.have_reached_retry_limit(latest_evaluated_goals=[g0, g0, g0, g1])
        is True
    )


def test_evaluated_goals_to_str() -> None:
    gs = [
        EvaluatedGoal(
            goal="foo0",
            motivation="bar0",
            completion_criteria="baz0",
            is_complete=False,
            reasoning="qux0",
            output=None,
        ),
        EvaluatedGoal(
            goal="foo1",
            motivation="bar1",
            completion_criteria="baz1",
            is_complete=True,
            reasoning="qux1",
            output="output",
        ),
    ]
    goals_str = GoalManager.evaluated_goals_to_str(gs)
    assert goals_str == (
        "## Goal 1:\n"
        "Goal: foo0\n"
        "Motivation: bar0\n"
        "Completion Criteria: baz0\n"
        "Is Complete: False\n"
        "Reasoning: qux0\n"
        "Output: None\n"
        "\n"
        "## Goal 2:\n"
        "Goal: foo1\n"
        "Motivation: bar1\n"
        "Completion Criteria: baz1\n"
        "Is Complete: True\n"
        "Reasoning: qux1\n"
        "Output: output\n"
    )


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
def test_generate_goal() -> None:
    goal_manager = GoalManager(
        agent_id="test_agent",
        high_level_description="You are a gambler that focuses on cycling races, predominantly the Tour de France.",
        agent_capabilities=(
            "- Web search\n"
            "- Web scraping\n"
            "- Accurate predictions of the probability of yes/no outcomes for a given event."
        ),
    )
    goal0 = goal_manager.generate_goal(latest_evaluated_goals=[])

    evaluated_goal = EvaluatedGoal(
        goal="Investigate the top 5 contenders for the Tour de France, make predictions on their chances of overall victory, and compare these against the market odds.",
        motivation="The Tour de France is a popular race, so markets are likely to have the highest liquidity",
        completion_criteria="5 contenders identified, predictions made, and compared against market odds",
        is_complete=False,
        reasoning="The Tour de France is cancelled this year.",
        output=None,
    )
    goal2 = goal_manager.generate_goal(latest_evaluated_goals=[evaluated_goal])

    # Generates a goal related to the Tour de France
    assert "Tour de France" in goal0.goal

    # Does not generate a goal related to the Tour de France, based on the
    # reasoning of the previous evaluated goal
    assert "Tour de France" not in goal2.goal


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
def test_evaluate_goal_progress() -> None:
    goal_manager = GoalManager(
        agent_id="test_agent",
        high_level_description="You are a gambler that focuses on cycling races, predominantly the Tour de France.",
        agent_capabilities=(
            "- Web search\n"
            "- Web scraping\n"
            "- Accurate predictions of the probability of yes/no outcomes for a given event."
        ),
    )
