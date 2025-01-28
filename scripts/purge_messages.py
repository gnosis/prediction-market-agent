import typer
from prediction_market_agent_tooling.gtypes import private_key_type, xdai_type

from prediction_market_agent.db.agent_communication import (
    fetch_count_unprocessed_transactions,
    pop_message,
)
from prediction_market_agent.utils import APIKeys


def main(private_key: str) -> None:
    keys = APIKeys(BET_FROM_PRIVATE_KEY=private_key_type(private_key))
    n_messages = fetch_count_unprocessed_transactions(
        consumer_address=keys.bet_from_address
    )

    if (
        input(
            f"Are you sure you want to purge all {n_messages} messages for agent {keys.bet_from_address}? (y/n): "
        )
        != "y"
    ):
        return

    popped = 0
    while fetch_count_unprocessed_transactions(consumer_address=keys.bet_from_address):
        pop_message(minimum_fee=xdai_type(0), api_keys=keys)
        popped += 1
        print(f"Popped {popped} messages.")


if __name__ == "__main__":
    typer.run(main)
