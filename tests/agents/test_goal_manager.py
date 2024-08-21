import pytest

from prediction_market_agent.agents.goal_manager import EvaluatedGoal, Goal, GoalManager
from prediction_market_agent.agents.microchain_agent.memory import (
    ChatHistory,
    ChatMessage,
)
from prediction_market_agent.utils import DEFAULT_OPENAI_MODEL
from tests.utils import RUN_PAID_TESTS

SQLITE_DB_URL = "sqlite://"


def test_have_reached_retry_limit() -> None:
    goal_manager = GoalManager(
        agent_id="test_agent",
        high_level_description="foo",
        agent_capabilities="bar",
        retry_limit=0,
        sqlalchemy_db_url=SQLITE_DB_URL,
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
        model=DEFAULT_OPENAI_MODEL,
        sqlalchemy_db_url=SQLITE_DB_URL,
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


def test_get_chat_history_after_goal_prompt() -> None:
    goal = Goal(goal="Foo", motivation="Bar", completion_criteria="Baz")
    assistant_message = ChatMessage(role="assistant", content="The answer is 42.")
    chat_history = ChatHistory(
        chat_messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content=goal.to_prompt()),
            assistant_message,
        ]
    )
    assert GoalManager.get_chat_history_after_goal_prompt(
        goal=goal, chat_history=chat_history
    ) == ChatHistory(chat_messages=[assistant_message])


def test_get_chat_history_after_goal_prompt_error() -> None:
    goal = Goal(goal="Foo", motivation="Bar", completion_criteria="Baz")
    assistant_message = ChatMessage(role="assistant", content="The answer is 42.")
    chat_history = ChatHistory(
        chat_messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
        ]
    )
    try:
        GoalManager.get_chat_history_after_goal_prompt(
            goal=goal, chat_history=chat_history
        )
    except ValueError as e:
        assert str(e) == "Goal prompt not found in chat history"


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
def test_evaluate_goal_progress_0() -> None:
    """
    Test for the case where the evaluated goal:
    - is completed
    - should have a 'None' output.
    """
    goal_manager = GoalManager(
        agent_id="",  # Not relevant to test
        high_level_description="",  # Not relevant to test
        agent_capabilities="",  # Not relevant to test
        model=DEFAULT_OPENAI_MODEL,
        sqlalchemy_db_url=SQLITE_DB_URL,
    )
    goal = Goal(
        goal="If last year's TdF winner is competing this year, place a small bet on them.",
        motivation="The winner of the last Tour de France is likely to be in good form.",
        completion_criteria="If the winner is competing, place a small bet, otherwise do nothing.",
    )
    chat_history0 = ChatHistory(
        chat_messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content=goal.to_prompt()),
            ChatMessage(
                role="assistant",
                content="Searching the web... Yes the winner, Tadej Pogacar, is competing.",
            ),
            ChatMessage(role="user", content="The reasoning has been recorded."),
            ChatMessage(
                role="assistant",
                content="The market id is '0x123' for the TdF winner. Placing bet of 0.01 USD on Tadej Pogacar",
            ),
            ChatMessage(role="user", content="Bet successfully placed."),
        ]
    )
    goal_evaluation = goal_manager.evaluate_goal_progress(
        goal=goal,
        chat_history=chat_history0,
    )
    assert goal_evaluation.is_complete is True
    assert goal_evaluation.output == None


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
def test_evaluate_goal_progress_1() -> None:
    """
    Test for the case where the evaluated goal:
    - is completed
    - should have a non-'None' output.
    """
    goal_manager = GoalManager(
        agent_id="",  # Not relevant to test
        high_level_description="",  # Not relevant to test
        agent_capabilities="",  # Not relevant to test
        model=DEFAULT_OPENAI_MODEL,
        sqlalchemy_db_url=SQLITE_DB_URL,
    )
    goal = Goal(
        goal="If last year's TdF winner is competing this year, get their probability of winning.",
        motivation="The winner of the last Tour de France is likely to be in good form.",
        completion_criteria="Return the name and odds of last year's winner for this year's TdF.",
    )
    chat_history0 = ChatHistory(
        chat_messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content=goal.to_prompt()),
            ChatMessage(
                role="assistant",
                content="Searching the web... Yes the winner, Tadej Pogacar, is competing. His winning probability: p_yes=0.27",
            ),
            ChatMessage(role="user", content="The reasoning has been recorded."),
        ]
    )
    goal_evaluation = goal_manager.evaluate_goal_progress(
        goal=goal,
        chat_history=chat_history0,
    )
    assert goal_evaluation.is_complete is True
    assert goal_evaluation.output is not None
    assert "Tadej Pogacar" in goal_evaluation.output
    assert "0.27" in goal_evaluation.output


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
def test_evaluate_goal_progress_2() -> None:
    """
    Test for the case where the evaluated goal is not completed
    """
    goal_manager = GoalManager(
        agent_id="",  # Not relevant to test
        high_level_description="",  # Not relevant to test
        agent_capabilities="",  # Not relevant to test
        model=DEFAULT_OPENAI_MODEL,
        sqlalchemy_db_url=SQLITE_DB_URL,
    )
    goal = Goal(
        goal="If last year's TdF winner is competing this year, get their probability of winning.",
        motivation="The winner of the last Tour de France is likely to be in good form.",
        completion_criteria="Return the name and odds of last year's winner for this year's TdF.",
    )
    chat_history0 = ChatHistory(
        chat_messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content=goal.to_prompt()),
            ChatMessage(
                role="assistant",
                content="Uhoh, I've hit some exception and need to quit",
            ),
        ]
    )
    goal_evaluation = goal_manager.evaluate_goal_progress(
        goal=goal,
        chat_history=chat_history0,
    )
    assert goal_evaluation.is_complete is False
    assert goal_evaluation.output == None
