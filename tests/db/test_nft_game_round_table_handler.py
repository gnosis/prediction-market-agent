from typing import Generator

import pytest
from freezegun import freeze_time

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.game_history import (
    DatetimeUTC,
    NFTGameRound,
    NFTGameRoundTableHandler,
)

GAME_ROUNDS = [
    NFTGameRound(
        id=1,
        start_time=DatetimeUTC(year=2025, month=10, day=1),
        end_time=DatetimeUTC(year=2025, month=10, day=3),
    ),
    NFTGameRound(
        id=2,
        start_time=DatetimeUTC(year=2025, month=10, day=4),
        end_time=DatetimeUTC(year=2025, month=10, day=6),
    ),
    NFTGameRound(
        id=3,
        start_time=DatetimeUTC(year=2025, month=10, day=7),
        end_time=DatetimeUTC(year=2025, month=10, day=9),
    ),
    NFTGameRound(
        id=4,
        start_time=DatetimeUTC(year=2025, month=10, day=10),
        end_time=DatetimeUTC(year=2025, month=10, day=12),
    ),
]


@pytest.fixture(scope="session")
def nft_game_round_table_handler() -> Generator[NFTGameRoundTableHandler, None, None]:
    """Creates a in-memory SQLite DB for testing"""
    prompt_handler = NFTGameRoundTableHandler(sqlalchemy_db_url="sqlite://")
    prompt_handler.sql_handler.save_multiple(GAME_ROUNDS)
    yield prompt_handler


@freeze_time(DatetimeUTC(year=2025, month=10, day=20))
def test_find_current_round_none(
    nft_game_round_table_handler: NFTGameRoundTableHandler,
) -> None:
    round_ = nft_game_round_table_handler.get_current_round()
    assert round_ is None


@freeze_time(DatetimeUTC(year=2025, month=10, day=5))
def test_find_current_round_success(
    nft_game_round_table_handler: NFTGameRoundTableHandler,
) -> None:
    round_ = nft_game_round_table_handler.get_current_round()
    assert round_ is not None
    assert round_.id == 2


@freeze_time(DatetimeUTC(year=2025, month=10, day=1))
def test_find_prev_round_none(
    nft_game_round_table_handler: NFTGameRoundTableHandler,
) -> None:
    round_ = nft_game_round_table_handler.get_previous_round()
    assert round_ is None


@freeze_time(DatetimeUTC(year=2025, month=10, day=5))
def test_find_prev_round_success(
    nft_game_round_table_handler: NFTGameRoundTableHandler,
) -> None:
    round_ = nft_game_round_table_handler.get_previous_round()
    assert round_ is not None
    assert round_.id == 1


@freeze_time(DatetimeUTC(year=2025, month=10, day=10))
def test_find_next_round_none(
    nft_game_round_table_handler: NFTGameRoundTableHandler,
) -> None:
    round_ = nft_game_round_table_handler.get_next_round()
    assert round_ is None


@freeze_time(DatetimeUTC(year=2025, month=10, day=5))
def test_find_next_round_success(
    nft_game_round_table_handler: NFTGameRoundTableHandler,
) -> None:
    round_ = nft_game_round_table_handler.get_next_round()
    assert round_ is not None
    assert round_.id == 3
