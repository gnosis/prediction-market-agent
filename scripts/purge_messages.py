import typer
from prediction_market_agent_tooling.gtypes import private_key_type

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.tools_nft_treasury_game import (
    purge_all_messages,
)
from prediction_market_agent.utils import APIKeys

APP = typer.Typer(pretty_exceptions_enable=False)


@APP.command()
def main(private_keys: list[str]) -> None:
    for private_key in private_keys:
        keys = APIKeys(BET_FROM_PRIVATE_KEY=private_key_type(private_key))
        purge_all_messages(keys)


if __name__ == "__main__":
    APP()
