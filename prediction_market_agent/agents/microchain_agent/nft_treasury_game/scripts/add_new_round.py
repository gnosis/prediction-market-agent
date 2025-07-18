from datetime import timedelta

import typer
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.utils import utcnow

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.game_history import (
    DatetimeUTC,
    NFTGameRound,
    NFTGameRoundTableHandler,
)


def main(
    from_: str,
    to_: str,
) -> None:
    now = utcnow()

    if from_ == "utcnow":
        start_time = now
    else:
        start_time = DatetimeUTC.to_datetime_utc(from_)

    if to_.isnumeric():
        end_time = start_time + timedelta(minutes=int(to_))
    else:
        end_time = DatetimeUTC.to_datetime_utc(to_)

    next_round = NFTGameRound(
        start_time=start_time,
        end_time=end_time,
    )

    if next_round.start_time < now:
        raise ValueError(f"Start time {next_round.start_time} must be in the future.")

    if (
        input(
            f"Current time is {now}. Are you sure you want to add the new round {next_round} (y/n)? "
        ).lower()
        != "y"
    ):
        logger.info("Operation cancelled.")
        return

    handler = NFTGameRoundTableHandler()
    handler.sql_handler.save_multiple([next_round])


if __name__ == "__main__":
    typer.run(main)
