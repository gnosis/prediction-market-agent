from pathlib import Path
from typing import Annotated

import typer

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.scripts.generate_report import (
    generate_report,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.scripts.reset_balance_anvil import (
    redistribute_nft_keys,
    reset_balances,
)

APP = typer.Typer(pretty_exceptions_enable=False)


@APP.command()
def main(
    rpc_url: str,
    output_dir: Path,
    xdai_balance_per_agent: Annotated[int, typer.Argument()] = 200,
    new_balance_treasury_xdai: Annotated[int, typer.Argument()] = 100,
) -> None:
    generate_report(
        rpc_url=rpc_url,
        output_dir=output_dir,
        initial_xdai_balance_per_agent=xdai_balance_per_agent,
    )
    reset_balances(
        rpc_url=rpc_url,
        new_balance_agents_xdai=xdai_balance_per_agent,
        new_balance_treasury_xdai=new_balance_treasury_xdai,
    )
    redistribute_nft_keys(rpc_url=rpc_url)


if __name__ == "__main__":
    APP()
