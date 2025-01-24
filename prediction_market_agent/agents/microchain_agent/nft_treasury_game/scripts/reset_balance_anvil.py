import typer

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    TREASURY_ADDRESS,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.deploy_nft_treasury_game import (
    DEPLOYED_NFT_AGENTS,
)
from prediction_market_agent.tools.anvil.anvil_requests import set_balance


def main(
    rpc_url: str, new_balance_agents_xdai: int, new_balance_treasury_xdai: int
) -> None:
    for agent in DEPLOYED_NFT_AGENTS:
        set_balance(
            rpc_url=rpc_url,
            address=agent.wallet_address,
            balance=new_balance_agents_xdai,
        )

    set_balance(
        rpc_url=rpc_url,
        address=TREASURY_ADDRESS,
        balance=new_balance_treasury_xdai,
    )


if __name__ == "__main__":
    typer.run(main)
