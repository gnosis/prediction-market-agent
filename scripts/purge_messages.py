import typer
from prediction_market_agent_tooling.gtypes import private_key_type, xdai_type

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.contracts import (
    AgentRegisterContract,
)
from prediction_market_agent.db.agent_communication import (
    fetch_count_unprocessed_transactions,
    pop_message,
)
from prediction_market_agent.utils import APIKeys

APP = typer.Typer(pretty_exceptions_enable=False)


@APP.command()
def main(private_keys: list[str]) -> None:
    register = AgentRegisterContract()

    for private_key in private_keys:
        keys = APIKeys(BET_FROM_PRIVATE_KEY=private_key_type(private_key))
        print(f"Purging messages for {keys.bet_from_address}.")

        register.register_as_agent(api_keys=keys)
        popped = 0
        while fetch_count_unprocessed_transactions(
            consumer_address=keys.bet_from_address
        ):
            pop_message(minimum_fee=xdai_type(0), api_keys=keys)
            popped += 1
            print(f"Popped {popped} messages.")
        register.deregister_as_agent(api_keys=keys)


if __name__ == "__main__":
    APP()
