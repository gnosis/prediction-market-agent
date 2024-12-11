import datetime
import typing as t

import pytest
from prediction_market_agent_tooling.tools.utils import utcnow
from sqlmodel import col

from prediction_market_agent.db.models import Prompt
from prediction_market_agent.db.sql_handler import SQLHandler


@pytest.fixture(scope="function")
def example_prompts() -> list[Prompt]:
    return [
        Prompt(prompt="prompt1", datetime_=utcnow(), session_identifier="a"),
        Prompt(
            prompt="prompt2",
            datetime_=utcnow() + datetime.timedelta(days=1),
            session_identifier="b",
        ),
    ]


def test_get_all(
    prompt_sql_handler_in_memory: SQLHandler, example_prompts: list[Prompt]
) -> None:
    p = example_prompts[0].model_copy()
    with prompt_sql_handler_in_memory.db_manager.get_session() as session:
        session.add(p)
        session.commit()
    assert len(prompt_sql_handler_in_memory.get_all()) == 1


def test_query_empty_database(prompt_sql_handler_in_memory: SQLHandler) -> None:
    # Assert: Verify the database is empty
    assert len(prompt_sql_handler_in_memory.get_all()) == 0


def get_first(
    sql_handler: SQLHandler, order_by_column_name: str, order_desc: bool
) -> Prompt | None:
    items: t.Sequence[Prompt] = sql_handler.get_with_filter_and_order(
        order_by_column_name=order_by_column_name, order_desc=order_desc, limit=1
    )
    return items[0] if items else None


def test_get_first(
    prompt_sql_handler_in_memory: SQLHandler, example_prompts: list[Prompt]
) -> None:
    prompt_earlier_date = example_prompts[0].prompt
    prompt_later_date = example_prompts[1].prompt
    column_to_order: str = Prompt.datetime_.key  # type: ignore[attr-defined]
    # insert 2 prompts with different dates
    prompt_sql_handler_in_memory.save_multiple(example_prompts)

    # Fetch latest prompt (desc)
    last_prompt = get_first(prompt_sql_handler_in_memory, column_to_order, True)
    assert last_prompt is not None
    # We assert on prompt str instead of referencing prompt object directly due to errors related to DetachedInstance
    # (see https://stackoverflow.com/questions/15397680/detaching-sqlalchemy-instance-so-no-refresh-happens)
    assert last_prompt.prompt == prompt_later_date

    # Fetch earliest prompt (asc)
    last_prompt = get_first(prompt_sql_handler_in_memory, column_to_order, False)
    assert last_prompt is not None
    assert last_prompt.prompt == prompt_earlier_date


def test_get_with_filter(
    prompt_sql_handler_in_memory: SQLHandler, example_prompts: list[Prompt]
) -> None:
    session_identifier = example_prompts[0].session_identifier
    prompt_sql_handler_in_memory.save_multiple(example_prompts)
    results: t.Sequence[
        Prompt
    ] = prompt_sql_handler_in_memory.get_with_filter_and_order(
        query_filters=[col(Prompt.session_identifier) == session_identifier]
    )
    assert len(results) == 1
    assert results[0].session_identifier == session_identifier
